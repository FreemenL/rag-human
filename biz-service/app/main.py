from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.responses import http_exception_handler
from app.routers import auth, claims, clauses, health, orders, plans, policies, premium, products


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="hm-insurance biz-service")
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1/auth")
    app.include_router(products.router, prefix="/api/v1/products")
    app.include_router(clauses.router, prefix="/api/v1/clauses")
    app.include_router(premium.router, prefix="/api/v1/premium")
    app.include_router(plans.router, prefix="/api/v1/insurance-plans")
    app.include_router(policies.router, prefix="/api/v1/policies")
    app.include_router(claims.router, prefix="/api/v1")
    app.include_router(orders.router, prefix="/api/v1/orders")
    app.state.started_at = datetime.now(timezone.utc)
    app.state.port = settings.server_port
    return app


app = create_app()
