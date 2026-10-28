"""BrandLens core: DB, Gemini client, RAG retrieval, rule checks, AI review pipeline."""
import os, re, json, sqlite3, hashlib, datetime as dt
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
client = genai.Client(api_key=API_KEY)

# ---------------------------------------------------------------- DB
DB_PATH = os.getenv("DB_PATH", "brandlens.db")

def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def hash_pw(p): return hashlib.pbkdf2_hmac("sha256", p.encode(), b"brandlens-salt", 100_000).hex()
def now(): return dt.datetime.utcnow().isoformat(timespec="seconds")

def init_db():
    c = db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, pw TEXT, role TEXT);
    CREATE TABLE IF NOT EXISTS knowledge(id INTEGER PRIMARY KEY, kind TEXT, title TEXT, text TEXT,
        owner TEXT, expiry TEXT, keywords TEXT, embedding TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS submissions(id INTEGER PRIMARY KEY, creator TEXT, channel TEXT, audience TEXT,
        status TEXT, risk TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS versions(id INTEGER PRIMARY KEY, submission_id INT, n INT, content TEXT, note TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS findings(id INTEGER PRIMARY KEY, version_id INT, origin TEXT, category TEXT,
        severity TEXT, confidence REAL, advisory INT, source_id INT, source_title TEXT, quote TEXT,
        issue TEXT, suggestion TEXT, status TEXT DEFAULT 'open', reason TEXT);
    CREATE TABLE IF NOT EXISTS comments(id INTEGER PRIMARY KEY, submission_id INT, user TEXT, action TEXT, text TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, ts TEXT, user TEXT, action TEXT, detail TEXT);
    """)
    for u, role in [("admin", "admin"), ("creator", "creator"), ("manager", "manager"), ("reviewer", "reviewer")]:
        c.execute("INSERT OR IGNORE INTO users(username,pw,role) VALUES(?,?,?)", (u, hash_pw(u + "123"), role))
    c.commit(); c.close()

def audit(user, action, detail=""):
    c = db(); c.execute("INSERT INTO audit(ts,user,action,detail) VALUES(?,?,?,?)", (now(), user, action, detail)); c.commit(); c.close()

# ---------------------------------------------------------------- Gemini helpers
def embed(texts, task="RETRIEVAL_DOCUMENT"):
    if not texts: return []
    out = []
    for i in range(0, len(texts), 50):
        r = client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i + 50],
                                        config=types.EmbedContentConfig(task_type=task))
        out += [e.values for e in r.embeddings]
    return out

def gen_json(prompt, system, parts=None):
    contents = (parts or []) + [prompt]
    try:
        r = client.models.generate_content(model=MODEL, contents=contents, config=types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", temperature=0.1))
        return json.loads(r.text)
    except Exception as e:
        print("Gemini error:", e)
        return None

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
def redact(text):
    """Strip emails/phones before any external AI call (privacy layer)."""
    return PHONE.sub("[PHONE]", EMAIL.sub("[EMAIL]", text))

# ---------------------------------------------------------------- Knowledge base
def add_knowledge(kind, title, text, owner="", expiry="", keywords=""):
    vec = embed([f"{title}\n{text}"])[0]
    c = db()
    cur = c.execute("INSERT INTO knowledge(kind,title,text,owner,expiry,keywords,embedding,created) VALUES(?,?,?,?,?,?,?,?)",
                    (kind, title, text, owner, expiry, keywords, json.dumps(vec), now()))
    c.commit(); i = cur.lastrowid; c.close(); return i

def extract_rules(guideline_text):
    """Gemini splits a guideline document into discrete, citable rules (admin reviews them afterwards)."""
    system = "You extract brand-guideline rules. Return ONLY JSON: [{\"title\": str, \"text\": str}]. One atomic rule each, no invention."
    return gen_json(guideline_text[:60000], system) or []

def retrieve(query_vecs, kinds=None, per_query=3, cap=14):
    c = db()
    rows = c.execute("SELECT * FROM knowledge").fetchall(); c.close()
    rows = [r for r in rows if (not kinds or r["kind"] in kinds)]
    if not rows or not query_vecs: return []
    M = np.array([json.loads(r["embedding"]) for r in rows]); M /= np.linalg.norm(M, axis=1, keepdims=True)
    picked = {}
    for q in query_vecs:
        q = np.array(q); q /= np.linalg.norm(q)
        sims = M @ q
        for idx in np.argsort(-sims)[:per_query]:
            picked[idx] = max(picked.get(idx, 0), float(sims[idx]))
    best = sorted(picked, key=lambda i: -picked[i])[:cap]
    return [rows[i] for i in best]

# ---------------------------------------------------------------- Deterministic (rule-based) checks
CLAIM_RE = re.compile(r"\b(best|#1|number one|no\.\s?1|guaranteed?|100%|world[- ]class|unbeatable|risk[- ]free|cure[sd]?|leading)\b", re.I)

def rule_checks(content):
    c = db()
    glossary = c.execute("SELECT * FROM knowledge WHERE kind='glossary'").fetchall()
    facts = c.execute("SELECT * FROM knowledge WHERE kind='fact'").fetchall(); c.close()
    out = []
    for g in glossary:  # glossary: title = banned term, text = preferred alternative / note
        if re.search(rf"\b{re.escape(g['title'])}\b", content, re.I):
            out.append(dict(origin="rule", category="terminology", severity="medium", confidence=1.0, advisory=0,
                            source_id=g["id"], source_title=f"Glossary: {g['title']}", quote=g["title"],
                            issue=f"'{g['title']}' is a banned/non-preferred term.", suggestion=g["text"]))
    for f in facts:  # expired facts whose keywords still appear
        if f["expiry"] and f["expiry"] < dt.date.today().isoformat():
            for kw in filter(None, (f["keywords"] or "").split(",")):
                if kw.strip() and kw.strip().lower() in content.lower():
                    out.append(dict(origin="rule", category="fact", severity="high", confidence=0.9, advisory=0,
                                    source_id=f["id"], source_title=f"Expired fact: {f['title']}", quote=kw.strip(),
                                    issue=f"Uses '{kw.strip()}', tied to a fact that expired on {f['expiry']}.",
                                    suggestion="Confirm the current figure with the fact owner: " + (f["owner"] or "unknown")))
    for m in CLAIM_RE.finditer(content):
        out.append(dict(origin="rule", category="claim", severity="medium", confidence=0.7, advisory=1,
                        source_id=None, source_title="Claim pattern", quote=m.group(0),
                        issue=f"Superlative/absolute claim '{m.group(0)}' needs substantiation.",
                        suggestion="Back it with an approved source or soften the wording."))
    return out

# ---------------------------------------------------------------- AI review (RAG + citation validation + judge)
REVIEW_SYSTEM = """You are a brand-governance reviewer. You may ONLY raise issues supported by the APPROVED SOURCES given.
Return ONLY JSON: [{"category":"tone|terminology|claim|fact|accessibility|visual","severity":"low|medium|high",
"confidence":0-1,"source_id":int|null,"quote":"exact text copied from the content","issue":"why it's a problem, plain language",
"suggestion":"minimal edit that preserves the author's meaning"}].
Rules: copy 'quote' verbatim from the content. Use source_id of the approved source you rely on. If a factual or promotional claim
cannot be verified from the sources, set source_id to null and say 'cannot verify' in the issue. Never invent rules. Return [] if all fine."""

JUDGE_SYSTEM = """You validate findings. For each finding decide if the cited source text genuinely supports it.
Return ONLY JSON: [{"index":int,"supported":bool}]. Findings with source_id null are 'cannot verify' advisories: mark supported true."""

def split_chunks(text):
    parts = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if len(p.strip()) > 20]
    return parts[:40] or [text[:500]]

def _validate_and_judge(raw, content, sources):
    by_id = {s["id"]: s for s in sources}
    kept = []
    for f in raw or []:
        sid = f.get("source_id")
        if sid is not None and sid not in by_id: continue                 # fake citation
        if f.get("quote") and f["quote"].lower() not in content.lower(): continue  # quote not in content
        kept.append(f)
    if kept:
        payload = [{"index": i, "issue": f["issue"], "quote": f["quote"],
                    "source_text": by_id[f["source_id"]]["text"] if f.get("source_id") in by_id else None}
                   for i, f in enumerate(kept)]
        verdict = gen_json(json.dumps(payload), JUDGE_SYSTEM)
        if verdict:
            ok = {v["index"] for v in verdict if v.get("supported")}
            kept = [f for i, f in enumerate(kept) if i in ok]
    out = []
    for f in kept:
        conf = float(f.get("confidence", 0.5)); sid = f.get("source_id")
        out.append(dict(origin="ai", category=f.get("category", "tone"), severity=f.get("severity", "low"),
                        confidence=conf, advisory=int(conf < 0.6 or sid is None), source_id=sid,
                        source_title=by_id[sid]["title"] if sid in by_id else "No approved source (cannot verify)",
                        quote=f.get("quote", ""), issue=f.get("issue", ""), suggestion=f.get("suggestion", "")))
    return out

def review_text(content, channel, audience):
    findings = rule_checks(content)
    safe = redact(content)
    sources = retrieve(embed(split_chunks(safe), "RETRIEVAL_QUERY"), kinds=["guideline", "fact"])
    if sources:
        src_txt = "\n".join(f"[{s['id']}] ({s['kind']}) {s['title']}: {s['text']}" for s in sources)
        prompt = f"CHANNEL: {channel}\nAUDIENCE: {audience}\n\nAPPROVED SOURCES:\n{src_txt}\n\nCONTENT:\n{safe}"
        findings += _validate_and_judge(gen_json(prompt, REVIEW_SYSTEM), safe, sources)
    return findings

def review_image(data: bytes, mime: str, channel):
    """Vision review: logo / palette / legibility / alt text, grounded in retrieved visual guidelines."""
    sources = retrieve(embed(["logo usage, colour palette, fonts, legibility, disclaimers, alt text"], "RETRIEVAL_QUERY"),
                       kinds=["guideline", "asset"], per_query=8)
    src_txt = "\n".join(f"[{s['id']}] {s['title']}: {s['text']}" for s in sources) or "(none)"
    prompt = (f"CHANNEL: {channel}\nAPPROVED SOURCES:\n{src_txt}\n\nReview the image. For 'quote' use a short description or visible text. "
              "Also add one finding with category 'accessibility' whose suggestion is proposed alt text.")
    raw = gen_json(prompt, REVIEW_SYSTEM, parts=[types.Part.from_bytes(data=data, mime_type=mime)])
    by_id = {s["id"]: s for s in sources}
    out = []
    for f in raw or []:
        sid = f.get("source_id")
        if sid is not None and sid not in by_id: continue
        conf = float(f.get("confidence", 0.5))
        out.append(dict(origin="ai", category=f.get("category", "visual"), severity=f.get("severity", "low"), confidence=conf,
                        advisory=int(conf < 0.6 or sid is None), source_id=sid,
                        source_title=by_id[sid]["title"] if sid in by_id else "No approved source (cannot verify)",
                        quote=f.get("quote", ""), issue=f.get("issue", ""), suggestion=f.get("suggestion", "")))
    return out

def risk_of(findings):
    return "high" if any(f["category"] in ("claim", "fact") and f["severity"] in ("medium", "high") for f in findings) else "low"
