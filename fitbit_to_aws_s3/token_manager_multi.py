# token_manager_multi.py

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

FITBIT_CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
FITBIT_CLIENT_SECRET = os.getenv("FITBIT_CLIENT_SECRET")
TOKEN_STORE_DIR = Path(os.getenv("TOKEN_STORE_DIR", "./secrets/tokens"))

logger = logging.getLogger(__name__)


def _get_token_path(user_id: str) -> Path:
    """Get token file path for a user."""
    return TOKEN_STORE_DIR / f"{user_id}.json"


def _ensure_token_dir() -> None:
    """Ensure token directory exists."""
    TOKEN_STORE_DIR.mkdir(parents=True, exist_ok=True)


def load_tokens_from_file(user_id: str) -> Optional[Dict]:
    """Load token data for a specific user, or None if not found."""
    path = _get_token_path(user_id)
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load tokens for {user_id}: {e}")
        return None


def save_tokens_to_file(user_id: str, token_data: Dict) -> None:
    """Save token data for a specific user."""
    _ensure_token_dir()
    path = _get_token_path(user_id)
    try:
        with open(path, "w") as f:
            json.dump(token_data, f, indent=4)
        logger.info(f"Saved tokens for {user_id}")
    except IOError as e:
        logger.error(f"Failed to save tokens for {user_id}: {e}")
        raise


def get_current_refresh_token(user_id: str) -> str:
    """Get the most recent refresh token for a user."""
    tokens = load_tokens_from_file(user_id)
    if tokens and "refresh_token" in tokens:
        return tokens["refresh_token"]

    raise RuntimeError(
        f"No refresh token found for user '{user_id}'. "
        f"Make sure they have gone through the Fitbit OAuth flow."
    )


def refresh_fitbit_tokens(user_id: str) -> Dict:
    """Use user's refresh token to get new access/refresh tokens from Fitbit."""
    if not FITBIT_CLIENT_ID or not FITBIT_CLIENT_SECRET:
        raise ValueError("FITBIT_CLIENT_ID or FITBIT_CLIENT_SECRET not set in .env")

    refresh_token = get_current_refresh_token(user_id)
    token_url = "https://api.fitbit.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    auth = HTTPBasicAuth(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET)

    try:
        resp = requests.post(token_url, data=data, auth=auth, timeout=10)
        resp.raise_for_status()
        token_data = resp.json()
        save_tokens_to_file(user_id, token_data)
        logger.info(f"Refreshed tokens for {user_id}")
        return token_data
    except requests.RequestException as e:
        logger.error(f"Token refresh failed for {user_id}: {e}")
        raise


def get_access_token(user_id: str) -> str:
    """Get fresh access token for a user (refreshing if needed)."""
    token_data = refresh_fitbit_tokens(user_id)
    return token_data.get("access_token")
