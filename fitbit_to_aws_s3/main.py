# main.py

import logging
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth

from token_manager_multi import save_tokens_to_file

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

app = FastAPI(title="Fitbit Multi-User OAuth")

FITBIT_CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
FITBIT_CLIENT_SECRET = os.getenv("FITBIT_CLIENT_SECRET")
FITBIT_REDIRECT_URI = os.getenv("FITBIT_REDIRECT_URI")
FITBIT_SCOPES = os.getenv("FITBIT_SCOPES", "")


@app.get("/")
def root() -> dict:
    """Health check endpoint."""
    return {
        "message": "Fitbit multi-user OAuth ready.",
        "hint": "Start onboarding with /fitbit/login?user_id=user_001",
    }


@app.get("/fitbit/login")
def fitbit_login(user_id: str) -> RedirectResponse:
    """Redirect user to Fitbit authorization."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id parameter required")

    safe_redirect = quote(FITBIT_REDIRECT_URI, safe="")
    scope_param = FITBIT_SCOPES.replace(" ", "+")

    authorize_url = (
        "https://www.fitbit.com/oauth2/authorize"
        f"?response_type=code"
        f"&client_id={FITBIT_CLIENT_ID}"
        f"&redirect_uri={safe_redirect}"
        f"&scope={scope_param}"
        f"&state={user_id}"
    )

    return RedirectResponse(authorize_url)


@app.get("/auth/callback")
def fitbit_callback(code: str | None = None, state: str | None = None) -> HTMLResponse:
    """Handle Fitbit OAuth callback."""
    if not code:
        return HTMLResponse(
            "<h3>Error: No authorization code from Fitbit</h3>",
            status_code=400
        )

    if not state:
        return HTMLResponse(
            "<h3>Error: No state (user_id) provided</h3>",
            status_code=400
        )

    user_id = state
    token_url = "https://api.fitbit.com/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": FITBIT_REDIRECT_URI,
    }
    auth = HTTPBasicAuth(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET)

    try:
        resp = requests.post(token_url, data=data, auth=auth, timeout=10)
        resp.raise_for_status()
        token_data = resp.json()
        save_tokens_to_file(user_id, token_data)
        logger.info(f"Successfully onboarded user: {user_id}")

        return HTMLResponse(
            f"<h2>✅ Fitbit OAuth Success</h2>"
            f"<p><b>User:</b> {user_id}</p>"
            f"<p>Tokens saved and ready for pipeline</p>"
        )
    except requests.RequestException as e:
        logger.error(f"Token exchange failed for {user_id}: {e}")
        return HTMLResponse(
            f"<h3>Error: {str(e)}</h3>",
            status_code=400
        )





