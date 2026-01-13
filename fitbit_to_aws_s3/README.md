# Fitbit Multi-User Data Pipeline to AWS S3

## Overview

Authenticates multiple Fitbit users via OAuth 2.0, fetches their health data, and uploads it to AWS S3.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env with your credentials
```

### 3. Start OAuth Server
```bash
python -m uvicorn main:app --reload --port 8000
```

### 4. Onboard Users
Visit: `http://localhost:8000/fitbit/login?user_id=user_001`

User authorizes → tokens saved to `secrets/tokens/user_001.json`

### 5. Run Pipeline
```bash
python fitbit_multi_to_s3.py
```

Fetches today's data for all users and uploads to S3.

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI OAuth server |
| `fitbit_multi_to_s3.py` | Fetch metrics & upload to S3 |
| `token_manager_multi.py` | Manage user tokens |

## Metrics Collected

- Steps (1-min)
- Calories (1-min)
- Active Zone Minutes (1-min)
- Heart Rate Variability (daily)
- Wrist Temperature (daily)

## S3 Structure

```
s3://fitbit-raw-data-hsr/users/{user_id}/{metric}/{filename}.csv
```

## Environment Variables

```env
FITBIT_CLIENT_ID=your_id
FITBIT_CLIENT_SECRET=your_secret
FITBIT_REDIRECT_URI=http://localhost:8000/auth/callback
TOKEN_STORE_DIR=./secrets/tokens
FITBIT_USER_IDS=user_001,user_002
AWS_DEFAULT_REGION=us-east-2
S3_BUCKET_RAW=your-bucket
AWS_PROFILE=your-profile
```

## How It Works

1. **OAuth** (`main.py`) - Users authorize via Fitbit, tokens stored securely
2. **Token Refresh** (`token_manager_multi.py`) - Keeps access tokens fresh
3. **Data Fetch** (`fitbit_multi_to_s3.py`) - Pulls metrics from Fitbit API
4. **CSV Save** - Stores data locally before upload
5. **S3 Upload** - Pushes files to AWS S3

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No refresh token | Complete OAuth flow at `/fitbit/login?user_id=...` |
| S3 upload fails | Check `AWS_PROFILE` has S3 permissions |
| API error 401 | Token will auto-refresh on next run |
| No data returned | User may lack permission or device has no data |

## Security

- `.env` and `secrets/tokens/` are gitignored
- AWS credentials via IAM profile (no hardcoded keys)
- Per-user token isolation