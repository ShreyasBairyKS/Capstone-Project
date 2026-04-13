# VisionFood QAI — Full-Stack Implementation Plan

**Date:** April 13, 2026  
**Model:** Claude Sonnet 4.6  
**Scope:** Cloud-hosted production system with MongoDB, RBAC, real-time WebSocket, product label ingestion with OCR, and race-condition-safe transactions.

---

## 1. Current State (What Already Exists)

| Component | Current | Verdict |
|---|---|---|
| Frontend | React 18 + TS + Tailwind + Vite | ✅ Keep as-is |
| API | FastAPI (Python) | ✅ Keep — do NOT switch to Flask (explained below) |
| Database | SQLite via SQLAlchemy + Alembic | ❌ Replace with MongoDB Atlas |
| Auth | Flat API Key header (`X-API-Key`) | ❌ Replace with JWT + RBAC |
| Real-time | WebSocket → Redis Stream | ✅ Keep architecture, enhance |
| Task queue | Celery + Redis | ✅ Keep |
| Inference | ONNX edge pipeline | ✅ Keep |
| Deployment | Docker Compose (local) | ❌ Migrate to cloud containers |

---

## 2. Why Keep FastAPI, Not Flask

The user asked about Flask. **Recommendation: stay with FastAPI.** Reasons:

- The entire codebase (1 800+ lines) already uses FastAPI async patterns, Pydantic v2 validation, and `asynccontextmanager` lifespan.
- FastAPI natively supports `async WebSocket` — Flask requires `flask-socketio` + `gevent` (a separate threading model that is harder to combine with async database drivers).
- `motor` (async MongoDB driver) integrates cleanly with FastAPI's event loop; it requires extra coordination with Flask.
- FastAPI Depends() gives a clean dependency injection point for RBAC decorators.
- **If Flask is a hard requirement**, use Quart (Flask API, fully async) with `quart-auth` and `motor`. The migration cost is ~3 days.

---

## 3. Recommended Middleware Stack

| Layer | Library | Purpose |
|---|---|---|
| Auth | `python-jose` + `passlib` | JWT sign/verify |
| RBAC | Custom `Depends()` factory | Role enforcement per route |
| Rate limit | `slowapi` (limits per user/IP) | Prevent abuse |
| CORS | `fastapi.middleware.cors` (already in) | Browser security |
| Audit log | `api/middleware/audit_logger.py` (already in) | Compliance trail |
| Metrics | `api/middleware/metrics.py` (Prometheus, already in) | Observability |
| Request ID | `asgi-correlation-id` | Trace requests end-to-end |

---

## 4. Database: MongoDB Atlas

### 4a. Why MongoDB Instead of PostgreSQL

- Schema-free documents map naturally to nested inspection results (detections, bounding boxes, severity, remediation — all in one document).
- MongoDB Atlas offers **multi-document ACID transactions** on replica sets (required for race condition safety).
- Atlas provides **Change Streams** — a native push mechanism to listen for new documents and feed them to WebSocket consumers without polling.
- Atlas free tier (M0) is sufficient for development; M10+ is required for transactions.

### 4b. Document Schema Design

```
inspections (collection)
├── _id: UUID string
├── product_id: str
├── sku: str
├── timestamp: ISODate
├── verdict: "PASS"|"FAIL"|"ESCALATE"|"REVIEW"
├── escalated: bool
├── latency_ms: float
├── device_id: str
├── attempt_count: int
├── __v: int                        ← optimistic concurrency version field
├── defects: [                      ← embedded, no join needed
│   { class_name, confidence, bbox, severity_grade, severity_score, uq_mean, uq_std }
│   ...
│ ]
└── remediation_action: {           ← embedded 1:1
      action, station, is_remediable, reason
    }

products (collection)
├── _id: UUID
├── sku: str (indexed, unique)
├── name: str
├── batch_id: str
├── label_info: {                   ← extracted from OCR / QR
│   qr_code: str,
│   expiry_date: str,
│   weight_g: float,
│   barcode: str,
│   raw_text: str
│ }
├── label_image_url: str            ← GCS/S3 path to uploaded image
├── label_pdf_url: str
├── created_at: ISODate
├── created_by: str (user_id)
└── __v: int

users (collection)
├── _id: UUID
├── email: str (unique index)
├── hashed_password: str
├── role: "viewer"|"operator"|"supervisor"|"admin"
├── is_active: bool
└── created_at: ISODate

audit_logs (collection)
├── _id: ObjectId
├── timestamp: ISODate
├── user_id: str
├── role: str
├── method: str
├── path: str
├── status_code: int
└── ip: str
```

### 4c. Python Driver

Use **Motor** (async MongoDB driver):

```
pip install motor==3.4.0
```

Replace `database/session.py` (SQLAlchemy engine) with:

```python
# database/session.py
from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings

_client: AsyncIOMotorClient | None = None

def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
    return _client

def get_db():
    return get_client()[settings.MONGODB_DB_NAME]
```

Add to `core/config.py`:

```python
MONGODB_URI: str = Field(default="mongodb+srv://user:pass@cluster.mongodb.net/")
MONGODB_DB_NAME: str = "visionfood"
```

---

## 5. RBAC Implementation

### 5a. Role Hierarchy

```
admin
  └─ supervisor
       └─ operator
            └─ viewer
```

Each role inherits all permissions of roles below it.

### 5b. Permission Matrix

| Endpoint | viewer | operator | supervisor | admin |
|---|---|---|---|---|
| GET /inspections | ✅ | ✅ | ✅ | ✅ |
| POST /inspections (submit image) | ❌ | ✅ | ✅ | ✅ |
| PATCH /inspections/{id}/verdict | ❌ | ✅ | ✅ | ✅ |
| GET /analytics | ✅ | ✅ | ✅ | ✅ |
| POST /products/register | ❌ | ✅ | ✅ | ✅ |
| GET /reports | ❌ | ❌ | ✅ | ✅ |
| POST /reports/generate | ❌ | ❌ | ✅ | ✅ |
| GET /admin/users | ❌ | ❌ | ❌ | ✅ |
| POST /admin/users | ❌ | ❌ | ❌ | ✅ |
| PATCH /admin/users/{id}/role | ❌ | ❌ | ❌ | ✅ |
| GET /models | ❌ | ❌ | ✅ | ✅ |
| PATCH /models/{id}/activate | ❌ | ❌ | ❌ | ✅ |

### 5c. JWT Flow

```
[Login] POST /auth/login
  body: { email, password }
  → returns { access_token (15 min), refresh_token (7 days) }

[Refresh] POST /auth/refresh
  header: Authorization: Bearer <refresh_token>
  → returns new access_token

[All other endpoints]
  header: Authorization: Bearer <access_token>
  → middleware decodes JWT, injects User + role into request state
```

### 5d. Code Pattern

```python
# api/dependencies.py additions

from enum import Enum

class Role(str, Enum):
    VIEWER     = "viewer"
    OPERATOR   = "operator"
    SUPERVISOR = "supervisor"
    ADMIN      = "admin"

_ROLE_RANK = {Role.VIEWER: 0, Role.OPERATOR: 1, Role.SUPERVISOR: 2, Role.ADMIN: 3}

def require_role(minimum: Role):
    """
    FastAPI Depends factory.
    Usage: Depends(require_role(Role.SUPERVISOR))
    """
    async def _check(token_data: TokenData = Depends(get_current_user)):
        if _ROLE_RANK[token_data.role] < _ROLE_RANK[minimum]:
            raise HTTPException(403, "Insufficient role")
        return token_data
    return _check

# In router:
@router.post("/reports/generate")
async def generate_report(
    user: TokenData = Depends(require_role(Role.SUPERVISOR)),
    ...
):
    ...
```

**Important:** Roles are enforced in code via `Depends()`, not just via login credentials. Even if a user obtains a valid JWT, the role embedded in the token (signed, tamper-proof) is checked on every request. The role can only be changed by an admin via `PATCH /admin/users/{id}/role`.

### 5e. Token Security

- Tokens signed with RS256 (asymmetric) — private key on server, public key can be distributed.
- Access token: 15-minute TTL.
- Refresh token stored as a `HttpOnly; Secure; SameSite=Strict` cookie to prevent XSS theft.
- Refresh token rotation on each use (old token invalidated in MongoDB `revoked_tokens` set).
- Token blacklist checked on protected endpoints (Redis set for speed).

---

## 6. Race Conditions & Transaction Control

### 6a. Problem Scenarios

1. **Two operators submit a verdict override on the same inspection simultaneously** — last write wins without transactions.
2. **Inspection is being written while a report is being generated** — partial data in report.
3. **Model activation** — two admins activate different model versions simultaneously.
4. **Product registration** — two lines register the same SKU concurrently.

### 6b. Solution: Optimistic Concurrency (Primary)

Every mutable document carries a `__v` (version) integer field.

```python
# Update pattern (Motor)
result = await db.inspections.find_one_and_update(
    {"_id": inspection_id, "__v": current_version},   # guard
    {"$set": {"verdict": new_verdict}, "$inc": {"__v": 1}},
    return_document=True,
)
if result is None:
    raise HTTPException(409, "Conflict: document was modified by another request. Reload and retry.")
```

If two users update the same document concurrently, exactly one succeeds and the other gets a 409 response with a clear message to retry. The frontend handles 409 by re-fetching, applying the change, and re-submitting.

### 6c. Solution: MongoDB Multi-Document Transactions (For Compound Writes)

Required when a single logical operation touches >1 collection (e.g., write inspection + update product defect counter + emit audit log atomically):

```python
async with await client.start_session() as session:
    async with session.start_transaction():
        await db.inspections.insert_one(doc, session=session)
        await db.products.update_one(
            {"sku": sku},
            {"$inc": {"total_inspections": 1, "defect_count": defect_delta}},
            session=session,
        )
        await db.audit_logs.insert_one(audit_doc, session=session)
        # Auto-commits on exit; rolls back on exception
```

**Requires MongoDB Atlas M10+ (replica set).**

### 6d. Solution: Redis Distributed Lock (For Critical Sections)

For operations where optimistic concurrency is impractical (e.g., model activation must be exclusive):

```python
from redis.asyncio import Redis
import uuid, asyncio

async def activate_model_exclusive(model_id: str, redis: Redis):
    lock_key = f"lock:model_activation"
    token = str(uuid.uuid4())
    acquired = await redis.set(lock_key, token, nx=True, ex=10)  # 10s max hold
    if not acquired:
        raise HTTPException(409, "Another model activation is in progress.")
    try:
        await perform_model_activation(model_id)
    finally:
        # Release only if we still own the lock (Lua script for atomicity)
        lua = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        await redis.eval(lua, 1, lock_key, token)
```

---

## 7. Real-Time Dashboard Updates — WebSocket + MongoDB Change Streams

### 7a. Is WebSocket the right choice?

**Yes.** The alternatives are:

| Approach | Pros | Cons |
|---|---|---|
| Polling (setInterval) | Simple | Unnecessary requests when no data; 1–5s lag |
| Server-Sent Events (SSE) | Simpler than WS | One-way only; no ping/pong; some proxy issues |
| **WebSocket** | **Bidirectional, low latency, persistent** | Slightly more complex setup |
| WebTransport | Future-proof | Not widely supported yet |

WebSocket is already implemented in `api/routers/websocket.py`. Keep it.

### 7b. Enhanced Architecture  

```
Production Line Camera
    │
    ▼
Edge Inference API (FastAPI)
    │ POST /inspections (writes inspection doc)
    ▼
MongoDB Atlas (inspections collection)
    │  ← Change Stream (watches insert/update events)
    ▼
Change Stream Listener (background asyncio task)
    │  publishes JSON to Redis Stream "inspections:live"
    ▼
WebSocket Router (already at /ws/live)
    │  fans out to all connected browser clients
    ▼
React Dashboard (useLiveInspections hook, already implemented)
```

The current code already has the Redis → WebSocket part. The **new component** is the MongoDB Change Stream listener that replaces any polling:

```python
# Add to api/main.py lifespan, after DB init:

async def _change_stream_listener(db, redis_client):
    """Watches MongoDB inspections for new documents and pushes to Redis Stream."""
    pipeline = [{"$match": {"operationType": {"$in": ["insert", "update"]}}}]
    async with db.inspections.watch(pipeline, full_document="updateLookup") as stream:
        async for change in stream:
            doc = change.get("fullDocument", {})
            # Serialize and push to the same Redis stream the WS router reads
            await redis_client.xadd(
                settings.REDIS_LIVE_STREAM,
                {"data": json.dumps(doc, default=str)},
                maxlen=1000,
            )

# Start as background task in lifespan:
asyncio.create_task(_change_stream_listener(db, redis))
```

### 7c. WebSocket Auth

Currently `/ws/live` has no auth. With RBAC, add token verification at connect time:

```python
@router.websocket("/ws/live")
async def live_inspection_stream(
    websocket: WebSocket,
    token: str = Query(...),   # ?token=<jwt>  passed by frontend
):
    user = verify_ws_token(token)   # raises if invalid/expired
    if _ROLE_RANK[user.role] < _ROLE_RANK[Role.VIEWER]:
        await websocket.close(code=4003)
        return
    await websocket.accept()
    ...
```

Frontend passes token: `new WebSocket('/ws/live?token=' + accessToken)`.

---

## 8. Product Input Interface + OCR Label Ingestion

### 8a. New Feature: `/products` Module

This is an entirely new module. It handles:
1. Operator manually enters product metadata (SKU, batch, name, weight).
2. Operator uploads a label image / PDF / provides a QR value.
3. OCR + QR engine extracts structured data.
4. Data is written to the `products` collection in MongoDB.
5. The `sku` field links inspections to product metadata.

### 8b. API Endpoints

```
POST   /products/register         — Submit product form + optional label file
GET    /products                  — List all products (paginated)
GET    /products/{sku}            — Get product by SKU
PATCH  /products/{sku}            — Update product metadata (operator+)
DELETE /products/{sku}            — Soft-delete (admin only)

POST   /products/extract-label    — Just run OCR/QR on a file, return structured JSON
                                    (no DB write — for preview before saving)
```

### 8c. Label Ingestion Pipeline

```
Input (one of):
  A) image file (JPEG/PNG of label)
  B) PDF file (product spec sheet)
  C) JSON payload (pre-structured from a barcode scanner)

         │
         ▼
┌─────────────────────────────────┐
│  LabelIngestionService          │
│                                 │
│  1. If JSON → validate directly │
│                                 │
│  2. If image →                  │
│     a. Decode with OpenCV       │
│     b. QR/barcode with pyzbar   │
│     c. OCR with pytesseract     │
│        (or EasyOCR for better   │
│         accuracy on curved      │
│         labels)                 │
│                                 │
│  3. If PDF →                    │
│     a. Extract text with        │
│        PyMuPDF (fitz)           │
│     b. Render page to image,    │
│        run QR decode            │
│     c. Parse structured fields  │
│        with regex rules         │
│                                 │
│  4. Normalize → LabelInfo schema│
└─────────────────────────────────┘
         │
         ▼
  products collection (MongoDB)
```

### 8d. OCR Library Choice

| Library | Speed | Accuracy | GPU | Notes |
|---|---|---|---|---|
| `pytesseract` | Fast | Medium | ❌ | Good for clean printed labels |
| `EasyOCR` | Slow | High | ✅ | Better for curved/rotated text |
| `paddleocr` | Medium | Very High | ✅ | Best for non-English + mixed |
| Google Vision API | Network | Highest | Cloud | Paid; no local processing needed |

**Recommendation:** Use `EasyOCR` for self-hosted (runs alongside inference container) + `pyzbar` for QR/barcode. If the label is always clean printed text, `pytesseract` is faster.

### 8e. New Pydantic Schema

```python
# core/schemas.py additions

class LabelExtractionInput(BaseModel):
    """Input to the label ingestion endpoint."""
    sku: str = Field(..., max_length=64)
    batch_id: Optional[str] = None
    # At least one of the following must be provided
    label_json: Optional[dict] = None
    # Files handled via UploadFile in the route, not here

class LabelInfo(BaseModel):
    qr_code: Optional[str] = None
    barcode: Optional[str] = None
    expiry_date: Optional[str] = None
    weight_g: Optional[float] = None
    product_name: Optional[str] = None
    raw_text: str = ""
    extraction_method: Literal["json", "qr", "ocr_image", "ocr_pdf"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

class ProductRecord(BaseModel):
    id: str
    sku: str
    name: Optional[str] = None
    batch_id: Optional[str] = None
    label_info: Optional[LabelInfo] = None
    label_image_url: Optional[str] = None
    created_at: datetime
    created_by: str
```

### 8f. File Upload & Storage

- Files POSTed as `multipart/form-data`.
- Store label images/PDFs in **Google Cloud Storage** (or AWS S3) — not in MongoDB (keeps documents small).
- Store just the GCS URL in the product document.
- Use signed URLs for upload (presigned PUT) so the browser uploads directly to GCS without going through the API server — reduces API bandwidth.

```python
# api/routers/products.py (new file)
from fastapi import APIRouter, Depends, File, Form, UploadFile

router = APIRouter(prefix="/products", tags=["Products"])

@router.post("/register", status_code=201)
async def register_product(
    sku: str = Form(...),
    batch_id: Optional[str] = Form(None),
    label_file: Optional[UploadFile] = File(None),
    label_json: Optional[str] = Form(None),
    user: TokenData = Depends(require_role(Role.OPERATOR)),
    db = Depends(get_db),
):
    label_info = await LabelIngestionService.extract(
        file=label_file,
        json_str=label_json,
    )
    doc = {
        "_id": str(uuid.uuid4()),
        "sku": sku,
        "batch_id": batch_id,
        "label_info": label_info.model_dump(),
        "created_by": user.user_id,
        "created_at": datetime.utcnow(),
        "__v": 0,
    }
    if label_file:
        url = await upload_to_gcs(label_file)
        doc["label_image_url"] = url

    await db.products.insert_one(doc)
    return doc
```

### 8g. Frontend: Product Registration Page

New tab in `TABS` array (`App.tsx`):

```typescript
{ id: 'product-input', label: 'Register Product', icon: Package, roles: ['operator', 'supervisor', 'admin'] }
```

New component `ProductInputForm.tsx`:
- SKU field (text input)
- Batch ID field
- File drop zone (accepts image/*, application/pdf)
- Preview pane: shows extracted label_info JSON
- "Extract & Preview" button → calls `POST /products/extract-label` (no save yet)
- "Save Product" button → calls `POST /products/register`
- QR code decoded result displayed as badge

---

## 9. Cloud Hosting Architecture

### 9a. Recommended Cloud: Google Cloud Platform (GCP)

GCP is recommended because:
- Cloud Run supports containerized FastAPI with auto-scaling to zero (cost-efficient).
- GCS for file storage (label images, PDFs, generated reports).
- Artifact Registry for Docker images.
- Cloud Armor for WAF / DDoS protection.
- Works natively with MongoDB Atlas (same region selection).

AWS or Azure are equally valid substitutes — the containers are portable.

### 9b. Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERNET                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS
                    ┌────▼─────┐
                    │Cloudflare│  CDN + WAF + DDoS protection
                    └────┬─────┘
                         │
           ┌─────────────┴──────────────┐
           │                            │
    ┌──────▼──────┐             ┌───────▼──────┐
    │  Vercel /   │             │  GCP Cloud   │
    │  Cloud Run  │             │  Run (API)   │
    │ (Dashboard) │             │  FastAPI     │
    │ React SPA   │             │  :8000       │
    └─────────────┘             └──────┬───────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼──────┐    ┌────────────▼────────┐  ┌──────────▼──────┐
    │  MongoDB Atlas │    │  Upstash Redis       │  │  GCS Bucket     │
    │  (M10 Replica) │    │  (WebSocket stream   │  │  (label imgs,   │
    │  Cloud DB      │    │   + distributed lock │  │   PDF, reports) │
    └────────────────┘    │   + token blacklist) │  └─────────────────┘
                          └──────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │  Production Line (On-Premise Edge Node)                  │
    │  - Edge FastAPI + ONNX inference (same codebase)        │
    │  - Camera capture loop                                   │
    │  - Pushes inspection results to Cloud API via HTTPS     │
    └─────────────────────────────────────────────────────────┘
```

### 9c. Cloud Run Service Configuration

```yaml
# cloud-run service definition (cloud/api-service.yaml)
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: visionfood-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"   # Always 1 warm instance for WS
        autoscaling.knative.dev/maxScale: "10"
    spec:
      timeoutSeconds: 3600   # Long timeout for WebSocket connections
      containers:
        - image: gcr.io/PROJECT_ID/visionfood-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: MONGODB_URI
              valueFrom:
                secretKeyRef:
                  name: mongodb-uri
                  key: latest
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: redis-url
                  key: latest
          resources:
            limits:
              memory: 2Gi
              cpu: "2"
```

**Note:** Cloud Run does support WebSocket — set `timeoutSeconds: 3600`.

### 9d. CI/CD Pipeline

```
GitHub Push to main
    │
    ▼  (GitHub Actions)
1. Run tests (pytest)
2. Build Docker image
    docker build -f docker/Dockerfile.api -t gcr.io/PROJECT_ID/visionfood-api:$SHA .
3. Push to Artifact Registry
4. Deploy to Cloud Run
    gcloud run deploy visionfood-api --image gcr.io/...
5. Build React (npm run build)
6. Deploy to Vercel (or Cloud Run static)
```

---

## 10. Phased Implementation Roadmap

### Phase 1 — Database Migration (Week 1–2)
- [ ] Sign up for MongoDB Atlas, create M10 cluster (us-central1 to match Cloud Run).
- [ ] Add `MONGODB_URI`, `MONGODB_DB_NAME` to `core/config.py`.
- [ ] Install `motor==3.4.0`, remove `sqlalchemy`, `alembic`.
- [ ] Rewrite `database/session.py` with Motor client.
- [ ] Rewrite `database/models.py` → `database/documents.py` (pure dicts + Pydantic validators).
- [ ] Rewrite `database/repositories/inspection_repository.py` for Motor (async find/insert/update).
- [ ] Create `database/repositories/product_repository.py`.
- [ ] Create `database/repositories/user_repository.py`.
- [ ] Add compound indexes in Atlas UI: `(sku, timestamp)`, `(verdict, timestamp)`.
- [ ] Update all router endpoints to use Motor repositories.
- [ ] Run existing tests against MongoDB (use `mongomock-motor` for unit tests).

### Phase 2 — Authentication & RBAC (Week 2–3)
- [ ] Install `python-jose[cryptography]`, `passlib[bcrypt]`.
- [ ] Generate RS256 key pair, store private key in GCP Secret Manager.
- [ ] Create `api/routers/auth.py`: `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout`.
- [ ] Create `api/dependencies.py` additions: `get_current_user`, `require_role()`.
- [ ] Create `api/routers/admin.py`: user management (CRUD), role assignment.
- [ ] Replace `APIKeyMiddleware` with JWT verification in `get_current_user`.
- [ ] Apply `Depends(require_role(...))` to every protected route per permission matrix (Section 5b).
- [ ] Frontend: add `LoginPage` (already in `App.tsx` imports), store JWT in memory + refresh token in HttpOnly cookie.
- [ ] Frontend: attach `Authorization: Bearer <token>` header in `api.ts` `apiFetch()`.
- [ ] Frontend: seed first admin user via a bootstrap script (`scripts/seed_admin.py`).
- [ ] Write auth unit tests (valid token, expired token, wrong role, missing role).

### Phase 3 — Product Input Interface + OCR (Week 3–4)
- [ ] Install `easyocr`, `pyzbar`, `pymupdf`, `python-multipart`.
- [ ] Create `core/ocr.py`: `LabelIngestionService` class.
- [ ] Create `api/routers/products.py` with endpoints from Section 8f.
- [ ] Create `database/repositories/product_repository.py`.
- [ ] Set up GCS bucket `visionfood-labels`, add `GOOGLE_APPLICATION_CREDENTIALS` to env.
- [ ] Create `core/storage.py`: `upload_to_gcs()`, `generate_signed_url()`.
- [ ] Frontend: create `src/components/ProductInputForm.tsx` (Section 8g).
- [ ] Add `product-input` tab to `App.tsx` with role guard.
- [ ] Integration test: upload a label image, verify `label_info` extracted correctly.

### Phase 4 — Real-time Enhancement (Week 4)
- [ ] Add MongoDB Change Stream listener as a background asyncio task in `api/main.py` lifespan.
- [ ] Enhance `/ws/live` to require JWT token query parameter (Section 7c).
- [ ] Frontend: pass JWT token when opening `WebSocket` in `useLiveInspections.ts`.
- [ ] Test: write an inspection, verify dashboard updates within 500ms.

### Phase 5 — Transaction Safety (Week 4–5)
- [ ] Add `__v` version field to all mutable documents.
- [ ] Implement optimistic concurrency in `inspection_repository.update_verdict()`.
- [ ] Implement compound transaction in inspection write path (Section 6c).
- [ ] Implement Redis distributed lock for model activation (Section 6d).
- [ ] Frontend: handle HTTP 409 responses — show "Reload and retry" toast, auto-retry once.
- [ ] Load test with `locust`: simulate 20 operators submitting verdicts on the same inspection simultaneously, verify no data corruption.

### Phase 6 — Cloud Deployment (Week 5–6)
- [ ] Create GCP project, enable Cloud Run, Artifact Registry, Secret Manager, GCS.
- [ ] Write `Dockerfile.api` (already exists) — verify it builds cleanly.
- [ ] Write `cloud/api-service.yaml` (Section 9c).
- [ ] Write `cloud/deploy.sh` or GitHub Actions workflow (Section 9d).
- [ ] Set up Cloudflare in front of Cloud Run for CDN + WAF.
- [ ] Move frontend env vars to Vercel environment settings.
- [ ] Configure CORS in `api/main.py` to allow the Vercel domain.
- [ ] Load test the cloud deployment before go-live.

---

## 11. New Dependencies Summary

```
# requirements.txt additions
motor==3.4.0                  # Async MongoDB driver
python-jose[cryptography]==3.3.0  # JWT
passlib[bcrypt]==1.7.4        # Password hashing
slowapi==0.1.9                # Rate limiting
asgi-correlation-id==4.3.0    # Request ID tracing
easyocr==1.7.1                # OCR for label images
pyzbar==0.1.9                 # QR/barcode decode
pymupdf==1.24.3               # PDF text extraction
python-multipart==0.0.9       # File upload support (already common)
google-cloud-storage==2.16.0  # GCS uploads

# requirements-dev.txt additions
mongomock-motor==0.0.21       # In-memory MongoDB for unit tests
locust==2.29.0                # Load testing
```

---

## 12. Security Checklist (OWASP Top 10)

| # | Risk | Mitigation |
|---|---|---|
| A01 | Broken Access Control | RBAC `require_role()` on every write/admin endpoint; roles in signed JWT |
| A02 | Cryptographic Failures | RS256 JWT; bcrypt passwords; HTTPS enforced by Cloudflare; HttpOnly cookies |
| A03 | Injection | Pydantic validation on all inputs; Motor parameterized queries (no raw string concat) |
| A04 | Insecure Design | Optimistic concurrency; distributed locks; transaction rollback on error |
| A05 | Security Misconfiguration | CORS restricted to known origins; no debug endpoints in production |
| A06 | Vulnerable Components | `pip audit` in CI; `npm audit` for dashboard |
| A07 | Auth Failures | Short-lived tokens (15 min); refresh token rotation; token blacklist in Redis |
| A08 | Data Integrity | MongoDB schema validation; Pydantic v2 on all API boundaries |
| A09 | Logging Failures | Audit log middleware already implemented; ship to Cloud Logging |
| A10 | SSRF | No user-controlled URLs fetched server-side; file uploads go direct to GCS |

---

## 13. Open Questions for Team Decision

1. **Flask vs FastAPI**: FastAPI is strongly recommended. If college requirement mandates Flask, use Quart instead.
2. **OCR hosting**: EasyOCR requires ~1.8GB RAM. Host on the same Cloud Run instance (set memory limit to 4GB) or as a separate `visionfood-ocr` Cloud Run service.
3. **Atlas tier**: M0 (free) for development, M10 ($57/month) for production (required for transactions + change streams).
4. **Upstash Redis vs GCP Memorystore**: Upstash is serverless (pay-per-request, free tier available) and easier. Memorystore is faster but $50+/month.
5. **Report generation (Celery)**: The existing Celery worker is fine. In cloud, run it as a separate Cloud Run Job triggered by a Pub/Sub message instead of a persistent worker — cheaper.
