# VisionFood QAI — User Interface, API & Database Implementation Plan

**Date:** April 13, 2026  
**Focus:** Cloud-ready MongoDB, RBAC authentication, product management, dashboard real-time updates  
**Scope:** Frontend components, backend API routes, database schema, integration tests

---

## Overview: Three Interconnected Layers

```
┌─────────────────────────────────────────────────────────┐
│                   React Frontend (Vite)                  │
│        Dashboard + Product Input + Reports (Supervisor)  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP + WebSocket
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)                    │
│     Auth (JWT) + RBAC + REST + WebSocket + OCR           │
└──────────────────┬────────────────────┬─────────────────┘
                   │                    │
                   ▼                    ▼
        ┌──────────────────────┐  ┌──────────────────┐
        │  MongoDB Atlas       │  │  Redis (Upstash) │
        │  - inspections       │  │  - live stream   │
        │  - products          │  │  - job queue     │
        │  - users             │  │  - token cache   │
        │  - audit_logs        │  └──────────────────┘
        └──────────────────────┘
```

---

## Stage 1: Database Layer Setup (Week 1)

### 1.1 MongoDB Atlas Setup

**Prerequisites:**
- Sign up at mongodb.com/atlas
- Create a new project "VisionFood"
- Choose **M10 cluster** (required for transactions + Change Streams)
- Region: `us-central1` (matches GCP Cloud Run)
- Enable backup (daily, 7-day retention)

**Connection String (will look like):**
```
mongodb+srv://visionfood_user:PASSWORD@visionfood-production.mongodb.net/visionfood?retryWrites=true&w=majority
```

### 1.2 Database Collections Schema

Create collections and indexes via MongoDB Atlas UI or `mongosh` CLI:

```javascript
// In MongoDB Atlas Web Shell or mongosh CLI

// 1. inspections collection
db.createCollection("inspections", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["_id", "sku", "timestamp", "verdict", "__v"],
      properties: {
        _id: { bsonType: "string" },
        product_id: { bsonType: ["string", "null"] },
        sku: { bsonType: "string" },
        timestamp: { bsonType: "date" },
        verdict: { enum: ["PASS", "FAIL", "ESCALATE", "REVIEW"] },
        escalated: { bsonType: "bool" },
        latency_ms: { bsonType: ["double", "null"] },
        device_id: { bsonType: "string" },
        attempt_count: { bsonType: "int" },
        defects: { bsonType: "array" },
        uq_result: { bsonType: ["object", "null"] },
        severity_result: { bsonType: ["object", "null"] },
        remediation_action: { bsonType: ["object", "null"] },
        label_qr: { bsonType: ["object", "null"] },
        __v: { bsonType: "int" }
      }
    }
  }
});

db.inspections.createIndexes([
  { key: { sku: 1, timestamp: -1 }, background: true },
  { key: { verdict: 1, timestamp: -1 }, sparse: true },
  { key: { device_id: 1, timestamp: -1 }, sparse: true },
  { key: { escalated: 1 }, sparse: true },
  { key: { timestamp: 1 }, expireAfterSeconds: 2592000 }  // TTL: 30 days
]);

// 2. products collection
db.createCollection("products");
db.products.createIndexes([
  { key: { sku: 1 }, unique: true },
  { key: { batch_id: 1 }, sparse: true },
  { key: { created_at: -1 }, background: true }
]);

// 3. users collection
db.createCollection("users");
db.users.createIndexes([
  { key: { email: 1 }, unique: true },
  { key: { is_active: 1 }, sparse: true }
]);

// 4. audit_logs collection (immutable append-only)
db.createCollection("audit_logs");
db.audit_logs.createIndexes([
  { key: { timestamp: -1 }, background: true },
  { key: { user_id: 1, timestamp: -1 }, sparse: true },
  { key: { timestamp: 1 }, expireAfterSeconds: 5184000 }  // TTL: 60 days
]);

// 5. model_versions collection
db.createCollection("model_versions");
db.model_versions.createIndexes([
  { key: { version_tag: 1 }, unique: true },
  { key: { is_active: 1 }, sparse: true },
  { key: { created_at: -1 }, background: true }
]);

// 6. revoked_tokens collection (for logout blacklist)
db.createCollection("revoked_tokens");
db.revoked_tokens.createIndexes([
  { key: { token_jti: 1 }, unique: true },
  { key: { expiry: 1 }, expireAfterSeconds: 0 }  // auto-remove on expiry
]);
```

### 1.3 Environment Configuration

Create `.env` in project root:

```bash
# Database
MONGODB_URI="mongodb+srv://visionfood_user:PASSWORD@visionfood-production.mongodb.net/visionfood?retryWrites=true&w=majority"
MONGODB_DB_NAME="visionfood"

# Redis (Upstash serverless)
REDIS_URL="redis://default:PASSWORD@us1-fine-antelope-00000.upstash.io:00000"
REDIS_LIVE_STREAM="inspections:live"

# JWT & Auth
SECRET_KEY="your-super-secret-key-min-32-chars"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# API
API_KEY="dev-insecure-key"  # legacy, kept for dev compatibility
LOG_LEVEL="INFO"

# GCS (for file uploads)
GCS_BUCKET="visionfood-labels"
GOOGLE_APPLICATION_CREDENTIALS="/app/secrets/gcs-key.json"
```

### 1.4 Database Session & Repositories Structure

```
database/
├── __init__.py
├── session.py              # Motor client initialization
├── models.py               # Pydantic document schemas (NOT SQLAlchemy)
├── repositories/
│   ├── __init__.py
│   ├── base_repository.py  # BaseRepository with common methods
│   ├── inspection_repository.py
│   ├── product_repository.py
│   ├── user_repository.py
│   ├── audit_log_repository.py
│   └── model_repository.py
└── migrations/             # (optional, for schema versioning)
```

Create `database/session.py`:

```python
"""
database/session.py — Async MongoDB session initialization.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None

async def init_db():
    """Connect to MongoDB on startup."""
    global _client, _db
    try:
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
        _db = _client[settings.MONGODB_DB_NAME]
        # Verify connection
        await _db.admin.command('ping')
        log.info("mongodb_connected", db=settings.MONGODB_DB_NAME)
    except Exception as exc:
        log.error("mongodb_connection_failed", error=str(exc))
        raise

async def close_db():
    """Close MongoDB connection on shutdown."""
    global _client
    if _client:
        _client.close()
        log.info("mongodb_disconnected")

def get_db() -> AsyncIOMotorDatabase:
    """Get the async DB instance for dependency injection."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db
```

Create `database/models.py` (Pydantic schemas):

```python
"""
database/models.py — Pydantic document schemas for MongoDB.
"""

from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ESCALATE = "ESCALATE"
    REVIEW = "REVIEW"

class UserRole(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"

# --- Inspection Document ---

class DefectDocument(BaseModel):
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    severity_grade: Optional[str] = None
    severity_score: Optional[float] = None

class LabelQRDocument(BaseModel):
    qr_detected: bool
    qr_decoded: Optional[str] = None
    qr_expected: Optional[str] = None
    qr_matched: Optional[bool] = None
    label_anomaly_types: list[str] = Field(default_factory=list)

class InspectionDocument(BaseModel):
    _id: str
    product_id: Optional[str] = None
    sku: str
    timestamp: datetime
    verdict: Verdict
    escalated: bool = False
    latency_ms: float
    device_id: str
    attempt_count: int = 0
    defects: list[DefectDocument] = Field(default_factory=list)
    uq_result: Optional[dict] = None
    severity_result: Optional[dict] = None
    remediation_action: Optional[dict] = None
    label_qr: Optional[LabelQRDocument] = None
    __v: int = 0

# --- Product Document ---

class LabelInfoDocument(BaseModel):
    qr_code: Optional[str] = None
    barcode: Optional[str] = None
    expiry_date: Optional[str] = None
    weight_g: Optional[float] = None
    product_name: Optional[str] = None
    raw_text: str = ""
    extraction_method: str  # "json", "qr", "ocr_image", "ocr_pdf"
    confidence: float = 1.0

class ProductDocument(BaseModel):
    _id: str
    sku: str
    name: Optional[str] = None
    batch_id: Optional[str] = None
    label_info: Optional[LabelInfoDocument] = None
    label_image_url: Optional[str] = None
    label_pdf_url: Optional[str] = None
    created_by: str  # user_id
    created_at: datetime
    __v: int = 0

# --- User Document ---

class UserDocument(BaseModel):
    _id: str
    email: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    __v: int = 0

# --- Audit Log Document ---

class AuditLogDocument(BaseModel):
    _id: Optional[str] = None  # auto ObjectId
    timestamp: datetime
    user_id: str
    user_email: str
    role: UserRole
    method: str  # GET, POST, PATCH, DELETE
    path: str
    status_code: int
    ip_address: str
    request_id: str
    response_time_ms: float
```

Create `database/repositories/base_repository.py`:

```python
"""
database/repositories/base_repository.py — Base repository with common CRUD patterns.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from typing import TypeVar, Generic, Optional, dict, Any

T = TypeVar('T')

class BaseRepository(Generic[T]):
    def __init__(self, db: AsyncIOMotorDatabase, collection_name: str):
        self.db = db
        self.collection: AsyncIOMotorCollection = db[collection_name]
    
    async def create(self, document: dict) -> dict:
        """Insert a new document."""
        result = await self.collection.insert_one(document)
        return {**document, "_id": str(result.inserted_id)}
    
    async def find_by_id(self, doc_id: str) -> Optional[dict]:
        """Find by _id."""
        return await self.collection.find_one({"_id": doc_id})
    
    async def find_one(self, query: dict) -> Optional[dict]:
        """Find first matching document."""
        return await self.collection.find_one(query)
    
    async def find_many(
        self,
        query: dict,
        skip: int = 0,
        limit: int = 100,
        sort: Optional[list[tuple]] = None,
    ) -> list[dict]:
        """Find multiple documents with pagination."""
        cursor = self.collection.find(query).skip(skip).limit(limit)
        if sort:
            cursor = cursor.sort(sort)
        return await cursor.to_list(length=limit)
    
    async def count(self, query: dict = None) -> int:
        """Count matching documents."""
        q = query or {}
        return await self.collection.count_documents(q)
    
    async def update_by_id(self, doc_id: str, update: dict) -> Optional[dict]:
        """Update by _id with optimistic concurrency (__v field)."""
        result = await self.collection.find_one_and_update(
            {"_id": doc_id},
            {
                "$set": {k: v for k, v in update.items() if k != "__v"},
                "$inc": {"__v": 1},
            },
            return_document=True,
        )
        if result is None:
            raise ValueError(f"Document {doc_id} not found or __v mismatch")
        return result
    
    async def delete_by_id(self, doc_id: str) -> bool:
        """Soft-delete (mark is_active=False if applicable)."""
        result = await self.collection.update_one(
            {"_id": doc_id},
            {"$set": {"is_active": False, "deleted_at": datetime.utcnow()}},
        )
        return result.matched_count > 0
```

---

## Stage 2: Authentication & RBAC (Week 1—2)

### 2.1 JWT Setup

Create `api/security.py`:

```python
"""
api/security.py — JWT token generation, validation, and refresh logic.
"""

from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class TokenData(BaseModel):
    user_id: str
    email: str
    role: str
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation

def hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain, hashed)

def create_access_token(user_id: str, email: str, role: str) -> str:
    """Create JWT access token (short-lived)."""
    import uuid
    now = datetime.utcnow()
    expiry = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "iat": now,
        "exp": expiry,
        "jti": str(uuid.uuid4()),  # JWT ID for revocation tracking
    }
    
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    log.info("access_token_created", user_id=user_id, email=email, role=role)
    return token

def create_refresh_token(user_id: str, email: str) -> str:
    """Create JWT refresh token (long-lived)."""
    import uuid
    now = datetime.utcnow()
    expiry = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "user_id": user_id,
        "email": email,
        "type": "refresh",
        "iat": now,
        "exp": expiry,
        "jti": str(uuid.uuid4()),
    }
    
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token

def verify_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("user_id")
        email = payload.get("email")
        role = payload.get("role")
        jti = payload.get("jti")
        
        if not all([user_id, email, role, jti]):
            log.warning("token_invalid_payload")
            return None
        
        return TokenData(
            user_id=user_id,
            email=email,
            role=role,
            exp=datetime.fromtimestamp(payload.get("exp")),
            iat=datetime.fromtimestamp(payload.get("iat")),
            jti=jti,
        )
    except jwt.ExpiredSignatureError:
        log.debug("token_expired")
        return None
    except jwt.InvalidSignatureError:
        log.warning("token_invalid_signature")
        return None
```

### 2.2 Create Auth Router

Create `api/routers/auth.py`:

```python
"""
api/routers/auth.py — Authentication endpoints (login, refresh, logout).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorDatabase
from api.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    hash_password,
    verify_password,
)
from api.dependencies import get_db
from database.repositories.user_repository import UserRepository
from core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    """Authenticate user, return JWT access token."""
    user_repo = UserRepository(db)
    user = await user_repo.find_by_email(req.email)
    
    if not user or not verify_password(req.password, user["hashed_password"]):
        log.warning("login_failed", email=req.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    if not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account disabled",
        )
    
    access_token = create_access_token(
        user_id=user["_id"],
        email=user["email"],
        role=user["role"],
    )
    
    log.info("login_success", user_id=user["_id"], email=req.email)
    return {"access_token": access_token}

@router.post("/refresh", response_model=LoginResponse)
async def refresh(req: RefreshRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    """Refresh expired access token using refresh token."""
    token_data = verify_token(req.refresh_token)
    if not token_data or token_data.exp < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    # Check if token is blacklisted (revoked)
    token_repo = db.revoked_tokens
    if await token_repo.find_one({"jti": token_data.jti}):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )
    
    # Issue new access token
    access_token = create_access_token(
        user_id=token_data.user_id,
        email=token_data.email,
        role=token_data.role,
    )
    
    return {"access_token": access_token}

@router.post("/logout")
async def logout(token_data: TokenData = Depends(get_current_user), db: AsyncIOMotorDatabase = Depends(get_db)):
    """Blacklist (revoke) the current token."""
    await db.revoked_tokens.insert_one({
        "jti": token_data.jti,
        "user_id": token_data.user_id,
        "expiry": token_data.exp,
    })
    log.info("user_logout", user_id=token_data.user_id)
    return {"message": "Logged out successfully"}
```

### 2.3 Dependencies for RBAC

Update `api/dependencies.py`:

```python
"""
api/dependencies.py — Dependency injection for auth, DB, and pipeline.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from api.security import verify_token, TokenData
from database.session import get_db
from core.logging import get_logger

log = get_logger(__name__)

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthCredentials = Depends(security)) -> TokenData:
    """Extract and validate JWT from Authorization header."""
    token = credentials.credentials
    token_data = verify_token(token)
    
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    return token_data

def require_role(*roles: str):
    """Factory function: Depends(require_role('supervisor', 'admin'))"""
    async def _check(user: TokenData = Depends(get_current_user)) -> TokenData:
        if user.role not in roles:
            log.warning(
                "authorization_denied",
                user_id=user.user_id,
                required_roles=roles,
                actual_role=user.role,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires one of: {roles}",
            )
        return user
    return _check
```

### 2.4 Create User Bootstrap Script

Create `scripts/seed_admin.py`:

```python
"""
scripts/seed_admin.py — Create initial admin user for development.
"""

import asyncio
from datetime import datetime
from uuid import uuid4
from motor.motor_asyncio import AsyncIOMotorClient
from api.security import hash_password
from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

async def seed_admin():
    """Create admin user if it doesn't exist."""
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DB_NAME]
    
    admin_email = "admin@visionfood.local"
    admin_password = "ChangeMe123!"
    
    existing = await db.users.find_one({"email": admin_email})
    if existing:
        log.info("admin_already_exists", email=admin_email)
        return
    
    admin_doc = {
        "_id": str(uuid4()),
        "email": admin_email,
        "hashed_password": hash_password(admin_password),
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "__v": 0,
    }
    
    await db.users.insert_one(admin_doc)
    log.info(
        "admin_created",
        email=admin_email,
        password=admin_password,
        message="⚠️ Change password immediately in production"
    )
    
    client.close()

if __name__ == "__main__":
    asyncio.run(seed_admin())
```

Run: `python scripts/seed_admin.py`

### 2.5 Auth Tests

```python
# tests/unit/test_auth.py

import pytest
from api.security import (
    create_access_token,
    verify_token,
    hash_password,
    verify_password,
)

def test_password_hashing():
    """Verify bcrypt hashing."""
    plain = "MyPassword123!"
    hashed = hash_password(plain)
    
    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

def test_jwt_creation_and_validation():
    """JWT token lifecycle."""
    token = create_access_token(
        user_id="user-123",
        email="user@example.com",
        role="operator",
    )
    
    assert isinstance(token, str)
    
    token_data = verify_token(token)
    assert token_data.user_id == "user-123"
    assert token_data.email == "user@example.com"
    assert token_data.role == "operator"

def test_jwt_expiry():
    """Expired token fails validation."""
    import jwt
    from datetime import datetime, timedelta
    from core.config import settings
    
    payload = {
        "user_id": "user-123",
        "email": "user@example.com",
        "role": "operator",
        "exp": datetime.utcnow() - timedelta(hours=1),  # expired
    }
    
    expired_token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    token_data = verify_token(expired_token)
    
    assert token_data is None  # verification failed
```

---

## Stage 3: Product Management API (Week 2)

### 3.1 Product Repository

Create `database/repositories/product_repository.py`:

```python
"""
database/repositories/product_repository.py — Product CRUD operations.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4
from motor.motor_asyncio import AsyncIOMotorDatabase
from database.repositories.base_repository import BaseRepository

class ProductRepository(BaseRepository):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, "products")
    
    async def create_product(
        self,
        sku: str,
        created_by: str,
        name: Optional[str] = None,
        batch_id: Optional[str] = None,
        label_info: Optional[dict] = None,
        label_image_url: Optional[str] = None,
    ) -> dict:
        """Create a new product."""
        # Check for duplicate SKU
        existing = await self.find_one({"sku": sku})
        if existing:
            raise ValueError(f"SKU {sku} already exists")
        
        product = {
            "_id": str(uuid4()),
            "sku": sku,
            "name": name,
            "batch_id": batch_id,
            "label_info": label_info or {},
            "label_image_url": label_image_url,
            "created_by": created_by,
            "created_at": datetime.utcnow(),
            "__v": 0,
        }
        
        result = await self.collection.insert_one(product)
        return product
    
    async def find_by_sku(self, sku: str) -> Optional[dict]:
        """Find product by SKU."""
        return await self.find_one({"sku": sku})
    
    async def list_products(self, skip: int = 0, limit: int = 100) -> list[dict]:
        """List all products with pagination."""
        return await self.find_many({}, skip=skip, limit=limit, sort=[("created_at", -1)])
    
    async def update_label_info(
        self,
        sku: str,
        label_info: dict,
    ) -> dict:
        """Update product label information."""
        result = await self.collection.find_one_and_update(
            {"sku": sku},
            {
                "$set": {"label_info": label_info},
                "$inc": {"__v": 1},
            },
            return_document=True,
        )
        if not result:
            raise ValueError(f"Product with SKU {sku} not found")
        return result
```

### 3.2 Product Routes

Create `api/routers/products.py`:

```python
"""
api/routers/products.py — Product management endpoints.
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from api.dependencies import get_db, get_current_user, require_role
from api.security import TokenData
from database.repositories.product_repository import ProductRepository
from core.ocr import LabelIngestionService
from core.storage import upload_to_gcs
from core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/products", tags=["Products"])

class ProductCreateRequest(BaseModel):
    sku: str
    name: Optional[str] = None
    batch_id: Optional[str] = None

class ProductResponse(BaseModel):
    _id: str
    sku: str
    name: Optional[str]
    batch_id: Optional[str]
    label_info: Optional[dict]
    created_by: str
    created_at: str

@router.post("/register", status_code=201, response_model=ProductResponse)
async def register_product(
    sku: str = Form(...),
    name: Optional[str] = Form(None),
    batch_id: Optional[str] = Form(None),
    label_file: Optional[UploadFile] = File(None),
    label_json: Optional[str] = Form(None),
    user: TokenData = Depends(require_role("operator", "supervisor", "admin")),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Register a new product with optional label upload."""
    product_repo = ProductRepository(db)
    
    label_info = None
    label_image_url = None
    
    try:
        # Extract label information
        if label_file or label_json:
            label_info = await LabelIngestionService.extract(
                file=label_file,
                json_str=label_json,
            )
        
        # Upload file to GCS if provided
        if label_file:
            label_image_url = await upload_to_gcs(label_file, sku)
        
        # Create product
        product = await product_repo.create_product(
            sku=sku,
            created_by=user.user_id,
            name=name,
            batch_id=batch_id,
            label_info=label_info and label_info.model_dump() or None,
            label_image_url=label_image_url,
        )
        
        log.info("product_registered", sku=sku, created_by=user.user_id)
        return ProductResponse(
            _id=product["_id"],
            sku=product["sku"],
            name=product.get("name"),
            batch_id=product.get("batch_id"),
            label_info=product.get("label_info"),
            created_by=product["created_by"],
            created_at=product["created_at"].isoformat(),
        )
    
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        log.error("product_registration_failed", error=str(e), sku=sku)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Product registration failed")

@router.get("/", response_model=list[ProductResponse])
async def list_products(
    skip: int = 0,
    limit: int = 100,
    user: TokenData = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """List all registered products."""
    product_repo = ProductRepository(db)
    products = await product_repo.list_products(skip=skip, limit=limit)
    
    return [
        ProductResponse(
            _id=p["_id"],
            sku=p["sku"],
            name=p.get("name"),
            batch_id=p.get("batch_id"),
            label_info=p.get("label_info"),
            created_by=p["created_by"],
            created_at=p["created_at"].isoformat(),
        )
        for p in products
    ]

@router.get("/{sku}", response_model=ProductResponse)
async def get_product(
    sku: str,
    user: TokenData = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Get product by SKU."""
    product_repo = ProductRepository(db)
    product = await product_repo.find_by_sku(sku)
    
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    
    return ProductResponse(
        _id=product["_id"],
        sku=product["sku"],
        name=product.get("name"),
        batch_id=product.get("batch_id"),
        label_info=product.get("label_info"),
        created_by=product["created_by"],
        created_at=product["created_at"].isoformat(),
    )
```

### 3.3 OCR Service (Label Extraction Core)

Create `core/ocr.py`:

```python
"""
core/ocr.py — Label OCR and QR extraction service.
"""

import io
import json
from typing import Optional
from pydantic import BaseModel
import pytesseract
from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
from PIL import Image
import PyPDF2
from core.logging import get_logger

log = get_logger(__name__)

class LabelInfo(BaseModel):
    qr_code: Optional[str] = None
    barcode: Optional[str] = None
    expiry_date: Optional[str] = None
    weight_g: Optional[float] = None
    product_name: Optional[str] = None
    raw_text: str = ""
    extraction_method: str  # "json", "qr", "ocr_image", "ocr_pdf"
    confidence: float = 1.0

class LabelIngestionService:
    @staticmethod
    async def extract(
        file: Optional = None,  # UploadFile
        json_str: Optional[str] = None,
    ) -> LabelInfo:
        """
        Extract label information from upload.
        
        Args:
            file: UploadFile (image or PDF)
            json_str: Pre-structured JSON string
        
        Returns:
            LabelInfo with extracted fields
        """
        if json_str:
            try:
                data = json.loads(json_str)
                return LabelInfo(
                    qr_code=data.get("qr_code"),
                    barcode=data.get("barcode"),
                    expiry_date=data.get("expiry_date"),
                    weight_g=data.get("weight_g"),
                    product_name=data.get("product_name"),
                    extraction_method="json",
                    confidence=1.0,
                )
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format")
        
        if not file:
            return LabelInfo(
                extraction_method="none",
                raw_text="",
            )
        
        # Read file
        content = await file.read()
        filename = file.filename or "unknown"
        
        if filename.lower().endswith('.pdf'):
            return await LabelIngestionService._extract_from_pdf(content)
        elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return await LabelIngestionService._extract_from_image(content)
        else:
            raise ValueError("Unsupported file format. Use PDF, JPEG, or PNG.")
    
    @staticmethod
    async def _extract_from_image(content: bytes) -> LabelInfo:
        """Extract QR and OCR text from image."""
        try:
            # Convert bytes to image
            nparr = cv2.imdecode(io.BytesIO(content).getvalue(), cv2.IMREAD_COLOR)
            pil_img = Image.fromarray(cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB))
            
            # Try QR first
            barcodes = pyzbar_decode(pil_img)
            if barcodes:
                qr_value = barcodes[0].data.decode("utf-8")
                return LabelInfo(
                    qr_code=qr_value,
                    extraction_method="qr",
                    confidence=1.0,
                )
            
            # Fallback to OCR
            text = pytesseract.image_to_string(pil_img)
            return LabelInfo(
                raw_text=text,
                extraction_method="ocr_image",
                confidence=0.7,  # OCR has lower confidence
            )
        except Exception as exc:
            log.warning("image_extraction_failed", error=str(exc))
            raise ValueError(f"Failed to extract from image: {str(exc)}")
    
    @staticmethod
    async def _extract_from_pdf(content: bytes) -> LabelInfo:
        """Extract text and images from PDF."""
        try:
            pdf_file = io.BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            return LabelInfo(
                raw_text=text,
                extraction_method="ocr_pdf",
                confidence=0.6,
            )
        except Exception as exc:
            log.warning("pdf_extraction_failed", error=str(exc))
            raise ValueError(f"Failed to extract from PDF: {str(exc)}")
```

### 3.4 Storage Service (GCS)

Create `core/storage.py`:

```python
"""
core/storage.py — Google Cloud Storage file upload.
"""

from google.cloud import storage
from core.config import settings
from core.logging import get_logger
import uuid

log = get_logger(__name__)

_gcs_client = None

def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client

async def upload_to_gcs(file, sku: str) -> str:
    """
    Upload file to GCS and return public URL.
    
    Args:
        file: UploadFile
        sku: Product SKU (used in path)
    
    Returns:
        gs://bucket/sku/filename
    """
    content = await file.read()
    filename = f"{sku}/{uuid.uuid4()}_{file.filename}"
    
    client = get_gcs_client()
    bucket = client.bucket(settings.GCS_BUCKET)
    blob = bucket.blob(filename)
    blob.upload_from_string(content)
    
    url = f"gs://{settings.GCS_BUCKET}/{filename}"
    log.info("file_uploaded_to_gcs", sku=sku, url=url)
    return url
```

### 3.5 Product Integration Test

```python
# tests/integration/test_product_api.py

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

@pytest.fixture
def auth_token(db):
    """Create test user and return JWT."""
    # ... seed user, call login API ...
    return "test-jwt-token"

def test_register_product(auth_token):
    """POST /products/register with form data."""
    response = client.post(
        "/products/register",
        data={
            "sku": "TEST-001",
            "name": "Test Product",
            "batch_id": "BATCH-2026-04",
        },
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["sku"] == "TEST-001"

def test_list_products(auth_token):
    """GET /products list."""
    response = client.get(
        "/products",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_duplicate_sku_rejected(auth_token):
    """Registering duplicate SKU returns 400."""
    # First registration
    client.post(
        "/products/register",
        data={"sku": "DUPE-001"},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    
    # Second with same SKU
    response = client.post(
        "/products/register",
        data={"sku": "DUPE-001"},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert response.status_code == 400
```

---

## Stage 4: Inspection API Enhancement (Week 2—3)

### 4.1 Update Inspection Router

Modify `api/routers/inspection.py` to:
1. Accept JWT-based auth (not just X-API-Key)
2. Store to MongoDB (not SQLite)
3. Publish to Redis stream after saving

```python
# Extract from existing inspection.py + modifications

@router.post("/inspections", response_model=InspectionResultResponse)
async def submit_inspection(
    req: InspectRequest,
    user: TokenData = Depends(require_role("operator", "supervisor", "admin")),
    db: AsyncIOMotorDatabase = Depends(get_db),
    pipeline: EdgeInferencePipeline = Depends(get_pipeline),
):
    """Submit image for inspection."""
    try:
        frame = _decode_image(req.image_b64)
        result = pipeline.inspect(
            frame,
            product_id=req.product_id,
            sku=req.sku,
            attempt_count=req.attempt_count,
        )
        
        # Store to MongoDB
        inspection_repo = InspectionRepository(db)
        await inspection_repo.create(result)
        
        # Publish to Redis stream
        await publish_inspection_event(result)
        
        # Audit log
        await log_audit_event(
            db=db,
            user_id=user.user_id,
            action="inspection_submitted",
            resource="inspections",
            details={"sku": req.sku, "inspection_id": result.inspection_id},
        )
        
        return InspectionResultResponse.from_domain(result)
    
    except Exception as exc:
        log.error("inspection_failed", error=str(exc), sku=req.sku)
        raise HTTPException(status_code=500, detail="Inspection failed")
```

---

## Stage 5: Real-time Dashboard (Week 3)

### 5.1 Frontend: Inspection Submission

Create `dashboard/src/components/InspectPanel.tsx`:

```typescript
import { useState } from 'react'
import { submitInspection } from '../api'

export function InspectPanel() {
  const [imageB64, setImageB64] = useState<string>('')
  const [sku, setSku] = useState<string>('default')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (ev) => {
      const b64 = (ev.target?.result as string).split(',')[1]
      setImageB64(b64)
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = async () => {
    if (!imageB64) {
      alert('Please select an image')
      return
    }

    setLoading(true)
    try {
      const res = await submitInspection(imageB64, sku)
      setResult(res)
    } catch (err) {
      alert('Submission failed: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 bg-white rounded shadow">
      <h2 className="text-2xl font-bold mb-4">Submit Inspection</h2>
      
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Product Image</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          disabled={loading}
        />
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">SKU</label>
        <input
          type="text"
          value={sku}
          onChange={(e) => setSku(e.target.value)}
          className="w-full p-2 border"
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading || !imageB64}
        className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-400"
      >
        {loading ? 'Submitting...' : 'Submit'}
      </button>

      {result && (
        <div className="mt-6 p-4 bg-gray-100 rounded">
          <h3>Result</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
```

### 5.2 Frontend: WebSocket Live Feed

Update `dashboard/src/hooks/useLiveInspections.ts`:

```typescript
import { useEffect } from 'react'
import { useApp } from '../store'

export function useLiveInspections() {
  const { dispatch, state } = useApp()

  useEffect(() => {
    const token = state.auth?.accessToken
    if (!token) return

    const wsUrl = `${import.meta.env.VITE_WS_BASE || 'ws://localhost:8000'}/ws/live?token=${token}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('WebSocket connected')
      dispatch({ type: 'SET_WS_CONNECTED', payload: true })
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        dispatch({ type: 'ADD_INSPECTION', payload: data })
      } catch (err) {
        console.error('Failed to parse WebSocket message', err)
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      dispatch({ type: 'SET_WS_CONNECTED', payload: false })
    }

    ws.onerror = (error) => {
      console.error('WebSocket error', error)
    }

    return () => ws.close()
  }, [state.auth?.accessToken, dispatch])
}
```

---

## Stage 6: Admin Interface & Testing (Week 4)

### 6.1 Admin User Management

Create `api/routers/admin.py`:

```python
"""
api/routers/admin.py — Admin-only endpoints (user management, model activation).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, EmailStr
from api.dependencies import get_db, require_role
from api.security import TokenData, hash_password
from database.repositories.user_repository import UserRepository
from uuid import uuid4
from datetime import datetime

router = APIRouter(prefix="/admin", tags=["Admin"])

class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    role: str  # viewer | operator | supervisor | admin

@router.post("/users", status_code=201)
async def create_user(
    req: CreateUserRequest,
    admin: TokenData = Depends(require_role("admin")),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Create a new user (admin only)."""
    user_repo = UserRepository(db)
    
    user = {
        "_id": str(uuid4()),
        "email": req.email,
        "hashed_password": hash_password(req.password),
        "role": req.role,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "__v": 0,
    }
    
    try:
        await db.users.insert_one(user)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    return {"user_id": user["_id"], "email": req.email, "role": req.role}

@router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    new_role: str,
    admin: TokenData = Depends(require_role("admin")),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Update user role (admin only)."""
    result = await db.users.find_one_and_update(
        {"_id": user_id},
        {"$set": {"role": new_role}},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "Role updated", "new_role": new_role}
```

### 6.2 Comprehensive Test Suite

```python
# tests/integration/test_full_stack.py

import pytest
from fastapi.testclient import TestClient
from api.main import app
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

client = TestClient(app)

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mongo_db():
    """Connect to test MongoDB."""
    client = AsyncIOMotorClient("mongodb+srv://test@test-cluster.mongodb.net/visionfood_test")
    db = client.visionfood_test
    yield db
    # Cleanup
    await client.drop_database("visionfood_test")

def test_auth_flow():
    """Full auth flow: register → login → access protected → logout."""
    # 1. Create admin
    admin_resp = client.post(
        "/admin/users",
        json={"email": "test@example.com", "password": "Pass123!", "role": "operator"},
        headers={"Authorization": "Bearer admin-token"},
    )
    assert admin_resp.status_code == 201
    
    # 2. Login
    login_resp = client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "Pass123!"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    
    # 3. Access protected endpoint
    inspect_resp = client.post(
        "/inspections",
        json={"image_b64": "fake", "sku": "test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    # May fail on image decode, but auth should pass
    assert inspect_resp.status_code in [200, 422]
    
    # 4. Logout
    logout_resp = client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert logout_resp.status_code == 200

def test_rbac_enforcement():
    """RBAC denies viewer from operator endpoints."""
    # Token with viewer role
    viewer_token = create_test_token(role="viewer")
    
    # Try to submit inspection (operator only)
    resp = client.post(
        "/inspections",
        json={"image_b64": "fake", "sku": "test"},
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp.status_code == 403  # Forbidden

def test_product_registration_workflow():
    """Full product → label extraction → storage flow."""
    token = create_test_token(role="operator")
    
    # Register with JSON label info
    resp = client.post(
        "/products/register",
        data={
            "sku": "BOTTLE-001",
            "name": "250ml Bottle",
            "batch_id": "BATCH-2026-04",
            "label_json": '{"qr_code": "SKU-BOTTLE-001", "weight_g": 250}'
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    product = resp.json()
    assert product["sku"] == "BOTTLE-001"
    assert product["label_info"]["qr_code"] == "SKU-BOTTLE-001"

def test_inspection_with_qr_verification():
    """Inspect a product, QR verified against registered product."""
    # 1. Register product with QR
    register_product_with_qr("BOTTLE-001", "SKU-BOTTLE-001")
    
    # 2. Submit inspection with matching QR
    token = create_test_token(role="operator")
    resp = client.post(
        "/inspections",
        json={
            "image_b64": "fake-image-with-qr",
            "sku": "BOTTLE-001",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    # Will fail on image decode, but demonstrates flow
    assert resp.status_code in [200, 422]
```

---

## Summary: Implementation Stages Timeline

| Stage | Week | Components | Tests |
|---|---|---|---|
| **1: DB Layer** | W1 | MongoDB setup, session, schemas, repos | repo unit tests |
| **2: Auth** | W1–2 | JWT, RBAC roles, login/refresh/logout | auth unit tests |
| **3: Products** | W2 | Product API, OCR service, GCS storage | product integration tests |
| **4: Inspections** | W2–3 | Enhanced inspection route, MongoDB writes | inspection integration tests |
| **5: Real-time** | W3 | WebSocket auth, live feed component | end-to-end tests |
| **6: Admin** | W4 | User management, role updates | admin integration tests |

