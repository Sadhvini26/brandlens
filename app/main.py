"""BrandLens API. Run: uvicorn app.main:app --reload"""
import os, io, jwt, datetime as dt
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .core import *

SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_TO_A_LONG_RANDOM_STRING")
app = FastAPI(title="BrandLens")
init_db()

# ---------------- auth / RBAC
def current_user(authorization: str = Header(None)):
    try:
        return jwt.decode(authorization.split()[1], SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Not signed in")

def need(*roles):
    def dep(u=Depends(current_user)):
        if u["role"] not in roles: raise HTTPException(403, "Your role cannot do this")
        return u
    return dep

class Login(BaseModel): username: str; password: str

@app.post("/api/login")
def login(b: Login):
    c = db(); u = c.execute("SELECT * FROM users WHERE username=?", (b.username,)).fetchone(); c.close()
    if not u or u["pw"] != hash_pw(b.password): raise HTTPException(401, "Wrong username or password")
    tok = jwt.encode({"sub": u["username"], "role": u["role"], "exp": dt.datetime.utcnow() + dt.timedelta(hours=12)}, SECRET)
    audit(u["username"], "login")
    return {"token": tok, "username": u["username"], "role": u["role"]}

# ---------------- file text extraction
def extract_text(name, data: bytes):
    n = name.lower()
    if n.endswith(".pdf"):
        import fitz
        return "\n".join(p.get_text() for p in fitz.open(stream=data, filetype="pdf"))
    if n.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")

# ---------------- knowledge base (admin)
@app.post("/api/knowledge/guideline")
async def import_guideline(file: UploadFile = File(None), text: str = Form(""), u=Depends(need("admin"))):
    raw = extract_text(file.filename, await file.read()) if file else text
    rules = extract_rules(raw)
    ids = [add_knowledge("guideline", r["title"], r["text"]) for r in rules]
    audit(u["sub"], "import_guideline", f"{len(ids)} rules")
    return {"rules_created": len(ids)}

class Item(BaseModel):
    kind: str            # fact | glossary | asset | guideline
    title: str           # glossary: the banned term
    text: str            # glossary: preferred alternative
    owner: str = ""; expiry: str = ""; keywords: str = ""

@app.post("/api/knowledge")
def add_item(i: Item, u=Depends(need("admin"))):
    kid = add_knowledge(i.kind, i.title, i.text, i.owner, i.expiry, i.keywords)
    audit(u["sub"], "add_knowledge", f"{i.kind}:{i.title}")
    return {"id": kid}

@app.get("/api/knowledge")
def list_knowledge(u=Depends(current_user)):
    c = db(); r = [dict(x) for x in c.execute("SELECT id,kind,title,text,owner,expiry FROM knowledge ORDER BY id DESC")]; c.close(); return r

@app.delete("/api/knowledge/{kid}")
def del_item(kid: int, u=Depends(need("admin"))):
    c = db(); c.execute("DELETE FROM knowledge WHERE id=?", (kid,)); c.commit(); c.close(); audit(u["sub"], "delete_knowledge", str(kid)); return {"ok": True}

# ---------------- submissions
def save_version(c, sid, content, note, findings):
    n = (c.execute("SELECT COALESCE(MAX(n),0) FROM versions WHERE submission_id=?", (sid,)).fetchone()[0]) + 1
    vid = c.execute("INSERT INTO versions(submission_id,n,content,note,created) VALUES(?,?,?,?,?)", (sid, n, content, note, now())).lastrowid
    for f in findings:
        c.execute("""INSERT INTO findings(version_id,origin,category,severity,confidence,advisory,source_id,source_title,quote,issue,suggestion)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (vid, f["origin"], f["category"], f["severity"], f["confidence"], f["advisory"],
                     f["source_id"], f["source_title"], f["quote"], f["issue"], f["suggestion"]))
    return vid

@app.post("/api/submissions")
async def create_submission(channel: str = Form("LinkedIn post"), audience: str = Form("General"), text: str = Form(""),
                            file: UploadFile = File(None), u=Depends(need("creator", "manager", "admin"))):
    findings, content = [], text
    if file and file.filename:
        data = await file.read()
        if len(data) > 10_000_000: raise HTTPException(413, "File too large (10 MB max)")
        if file.content_type and file.content_type.startswith("image/"):
            content = f"[image: {file.filename}]"
            findings = review_image(data, file.content_type, channel)
        else:
            content = extract_text(file.filename, data); findings = review_text(content, channel, audience)
    elif text.strip():
        findings = review_text(text, channel, audience)
    else:
        raise HTTPException(400, "Provide text or a file")
    c = db()
    sid = c.execute("INSERT INTO submissions(creator,channel,audience,status,risk,created) VALUES(?,?,?,?,?,?)",
                    (u["sub"], channel, audience, "draft", risk_of(findings), now())).lastrowid
    save_version(c, sid, content, "Initial AI review", findings); c.commit(); c.close()
    audit(u["sub"], "create_submission", str(sid))
    return {"id": sid}

@app.get("/api/submissions")
def list_subs(u=Depends(current_user)):
    c = db(); q = "SELECT * FROM submissions" + (" WHERE creator=?" if u["role"] == "creator" else "") + " ORDER BY id DESC"
    r = [dict(x) for x in c.execute(q, (u["sub"],) if u["role"] == "creator" else ())]; c.close(); return r

@app.get("/api/submissions/{sid}")
def get_sub(sid: int, u=Depends(current_user)):
    c = db(); s = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
    if not s: raise HTTPException(404)
    if u["role"] == "creator" and s["creator"] != u["sub"]: raise HTTPException(403)
    vs = [dict(v) for v in c.execute("SELECT * FROM versions WHERE submission_id=? ORDER BY n", (sid,))]
    for v in vs: v["findings"] = [dict(f) for f in c.execute("SELECT * FROM findings WHERE version_id=?", (v["id"],))]
    cm = [dict(x) for x in c.execute("SELECT * FROM comments WHERE submission_id=? ORDER BY id", (sid,))]; c.close()
    return {**dict(s), "versions": vs, "comments": cm}

class Resolve(BaseModel): action: str; reason: str = ""; replacement: str = ""   # accept | edit | dismiss

@app.post("/api/findings/{fid}/resolve")
def resolve(fid: int, b: Resolve, u=Depends(need("creator", "manager", "admin"))):
    c = db(); f = c.execute("SELECT * FROM findings WHERE id=?", (fid,)).fetchone()
    if not f: raise HTTPException(404)
    if b.action == "dismiss":
        if not b.reason.strip(): raise HTTPException(400, "A reason is required to dismiss a finding")
        c.execute("UPDATE findings SET status='dismissed', reason=? WHERE id=?", (b.reason, fid))
    elif b.action in ("accept", "edit"):
        new = b.replacement if b.action == "edit" else f["suggestion"]
        v = c.execute("SELECT * FROM versions WHERE id=?", (f["version_id"],)).fetchone()
        content = v["content"].replace(f["quote"], new, 1) if f["quote"] else v["content"]
        c.execute("UPDATE findings SET status=? WHERE id=?", ("accepted" if b.action == "accept" else "edited", fid))
        save_version(c, v["submission_id"], content, f"{b.action}ed: {f['issue'][:60]}", [])
    else: raise HTTPException(400, "Unknown action")
    c.commit(); c.close(); audit(u["sub"], f"finding_{b.action}", str(fid)); return {"ok": True}

@app.post("/api/submissions/{sid}/recheck")
def recheck(sid: int, u=Depends(need("creator", "manager", "admin"))):
    c = db(); s = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
    v = c.execute("SELECT * FROM versions WHERE submission_id=? ORDER BY n DESC LIMIT 1", (sid,)).fetchone()
    fs = review_text(v["content"], s["channel"], s["audience"])
    save_version(c, sid, v["content"], "Re-check", fs)
    c.execute("UPDATE submissions SET risk=? WHERE id=?", (risk_of(fs), sid)); c.commit(); c.close(); return {"ok": True}

@app.post("/api/submissions/{sid}/submit")
def submit(sid: int, u=Depends(need("creator", "manager", "admin"))):
    c = db(); s = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
    status = "pending_reviewer" if s["risk"] == "high" else "pending_manager"   # risk-based routing
    c.execute("UPDATE submissions SET status=? WHERE id=?", (status, sid)); c.commit(); c.close()
    audit(u["sub"], "submit_for_review", f"{sid}->{status}"); return {"status": status}

class Decision(BaseModel): decision: str; comment: str = ""    # approve | changes | reject

@app.post("/api/submissions/{sid}/decide")
def decide(sid: int, b: Decision, u=Depends(need("manager", "reviewer", "admin"))):
    c = db(); s = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
    if s["status"] == "pending_reviewer" and u["role"] not in ("reviewer", "admin"):
        raise HTTPException(403, "High-risk content needs an authorised reviewer")
    status = {"approve": "cleared_for_publication", "changes": "changes_requested", "reject": "rejected"}[b.decision]
    c.execute("UPDATE submissions SET status=? WHERE id=?", (status, sid))
    c.execute("INSERT INTO comments(submission_id,user,action,text,created) VALUES(?,?,?,?,?)", (sid, u["sub"], b.decision, b.comment, now()))
    c.commit(); c.close(); audit(u["sub"], f"decision_{b.decision}", str(sid)); return {"status": status}

# ---------------- insights & audit
@app.get("/api/insights")
def insights(u=Depends(need("manager", "admin"))):
    c = db()
    cats = [dict(r) for r in c.execute("SELECT category, COUNT(*) n FROM findings GROUP BY category ORDER BY n DESC")]
    dism = [dict(r) for r in c.execute("""SELECT source_title, COUNT(*) total, SUM(status='dismissed') dismissed FROM findings
        WHERE source_id IS NOT NULL GROUP BY source_id HAVING total>=1 ORDER BY dismissed DESC, total DESC LIMIT 10""")]
    c.close()
    summary = None
    if any(d["dismissed"] for d in dism):
        summary = gen_json(f"Rules often dismissed by creators: {dism}. For the top 3, suggest why the rule may be unclear and how to reword it. "
                           'Return JSON [{"rule":str,"likely_problem":str,"suggested_rewrite":str}].',
                           "You help brand owners debug unclear guidelines. Return only JSON.")
    return {"by_category": cats, "most_dismissed_rules": dism, "unclear_rule_summary": summary}

@app.get("/api/audit")
def get_audit(u=Depends(need("admin"))):
    c = db(); r = [dict(x) for x in c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT 200")]; c.close(); return r

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
@app.get("/")
def index(): return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
