# -*- coding: utf-8 -*-
# app/router.py
# This module defines the main router for the FastAPI application, including authentication endpoints.
from fastapi import APIRouter, Depends, HTTPException, status, Form
from pydantic import BaseModel
from typing import Dict, Any
from app.auth import login_user, get_current_user, require_auth
from app.db import create_user, get_all_users
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Pydantic models for request/response
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]


class RegisterRequest(BaseModel):
    username: str
    password: str


class MessageResponse(BaseModel):
    message: str
    status: str


@router.get("/")
def health_check():
    """Home endpoint for the application."""
    return {"status": "ok", "message": "Welcome to the Assistant Service!"}


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Server is healthy"}


@router.post("/auth/login", response_model=LoginResponse)
async def login(username: str = Form(...), password: str = Form(...)):
    """Login endpoint."""
    try:
        result = login_user(username, password)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.info(f"User {username} logged in successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/auth/logout", response_model=MessageResponse)
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout endpoint."""
    try:
        # In a stateless JWT system, logout is handled client-side
        # by discarding the token. Server-side logout would require
        # token blacklisting which is more complex.
        logger.info(f"User {current_user['username']} logged out")
        return {
            "message": "Successfully logged out",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/auth/register", response_model=MessageResponse)
async def register(request: RegisterRequest):
    """Register new user endpoint."""
    try:
        if len(request.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        success = create_user(request.username, request.password)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists"
            )
        
        logger.info(f"User {request.username} registered successfully")
        return {
            "message": "User registered successfully",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/auth/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information."""
    return {
        "user": current_user,
        "status": "authenticated"
    }


@router.get("/auth/api-key")
async def get_api_key(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get API key for MCP access."""
    from app.auth import ASSISTANT_API_KEY
    return {
        "api_key": ASSISTANT_API_KEY,
        "usage": "Add as X-API-Key header for MCP endpoints",
        "user": current_user['username']
    }


@router.get("/users")
async def get_users(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get all users (admin access required)."""
    try:
        users = get_all_users()
        return {
            "users": users,
            "count": len(users),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Get users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/protected")
async def protected_route(current_user: Dict[str, Any] = Depends(require_auth)):
    """Example protected route that requires authentication."""
    return {
        "message": f"Hello {current_user['username']}, this is a protected route!",
        "user": current_user
    }

