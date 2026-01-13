# Fitbit Multi-User Data Pipeline to AWS S3

## What This Does

Authenticates multiple Fitbit users via OAuth 2.0, fetches their health data (steps, calories, HRV, etc.), and uploads it to AWS S3.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env with your Fitbit credentials and AWS profile
```

### 3. Start OAuth Server
```bash
python -m uvicorn main:app --reload --port 8000
```

### 4. Onboard a User
Visit: `http://localhost:8000/fitbit/login?user_id=user_001`

User authorizes → tokens saved to `secrets/tokens/user_001.json`

### 5. Run Data Pipeline
```bash
python fitbit_multi_to_s3.py
```

Fetches today's data for all users in `FITBIT_USER_IDS` and uploads to S3.

## Project Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI OAuth server |
| `fitbit_multi_to_s3.py` | Fetch data & upload to S3 |
| `token_manager_multi.py` | Manage user tokens |
| `.env` | Configuration (do not commit) |
| `.env.example` | Configuration template |

## Metrics Collected

- Steps (1-min intervals)
- Calories (1-min intervals)
- Active Zone Minutes (1-min intervals)
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
S3_BUCKET_RAW=your-bucket-name
AWS_PROFILE=your-aws-profile
```

## Security

- `.env` and `secrets/tokens/` are gitignored
- AWS credentials via IAM profile (no hardcoded keys)
- Each user has isolated token storage

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No refresh token found" | User hasn't completed OAuth flow at `/fitbit/login?user_id=...` |
| S3 upload fails | Check `AWS_PROFILE` and S3 bucket permissions |
| API returns 401 | Token refresh will happen automatically |