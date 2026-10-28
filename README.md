# BrandLens (Round 2 MVP)
RAG-based brand governance with human-in-the-loop review. Gemini is the only AI API used.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then paste your GEMINI_API_KEY
uvicorn app.main:app --reload
```
Open http://localhost:8000. Demo logins: admin / creator / manager / reviewer (password = name + `123`). Change these before deploying.

## Try it
1. Sign in as **admin** → Knowledge base → paste guidelines (or upload PDF/DOCX) → Extract rules.
2. Add a glossary term (title `cheap`, text `affordable`), and an approved fact with an expiry date and keywords.
3. Sign in as **creator** → paste a draft ("We are the #1 cheap CRM, guaranteed!") → Run review.
4. Accept / edit / dismiss (reason required) → Send for approval. High-risk (claim/fact) content routes to **reviewer**, others to **manager**.

## Mapping to the plan
- `core.py` → RAG (Gemini embeddings + cosine retrieval), rule checks, citation validation, AI judge, redaction, image review.
- `main.py` → RBAC (JWT), workflow, versioning, audit log, insights.
- Storage is SQLite with embeddings as JSON for zero-setup; swap to PostgreSQL + pgvector in `core.py` (`retrieve`, `db`) for deployment.
- Frontend is a dependency-free single page; port to React + TypeScript + Tailwind later if needed.
