# fitbit_multi_to_s3.py

import logging
import os
from pathlib import Path
from datetime import date
from typing import Dict

import pandas as pd
import boto3
import requests
from dotenv import load_dotenv

from token_manager_multi import get_access_token

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
S3_BUCKET_RAW = os.getenv("S3_BUCKET_RAW")

USER_IDS_RAW = os.getenv("FITBIT_USER_IDS", "")
USER_IDS = [u.strip() for u in USER_IDS_RAW.split(",") if u.strip()]

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.Session(
        profile_name=AWS_PROFILE,
        region_name=AWS_REGION,
    ).client("s3")


def save_df_to_csv(df: pd.DataFrame, user_id: str, prefix: str) -> Path:
    """Save DataFrame to CSV file."""
    if df.empty:
        logger.warning(f"Empty DataFrame for {user_id}/{prefix}")
        return None

    timestamp = date.today().strftime("%Y%m%d")
    filename = f"{user_id}_{prefix}_{timestamp}.csv"
    filepath = DATA_DIR / filename

    df.to_csv(filepath, index=False)
    logger.info(f"Saved: {filepath}")
    return filepath


def upload_file_to_s3(
    file_path: Path,
    user_id: str,
    metric: str,
    s3_key_prefix_root: str = "users",
) -> str:
    """Upload file to S3."""
    if not file_path or not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None

    s3_key = f"{s3_key_prefix_root}/{user_id}/{metric}/{file_path.name}"
    s3_client = get_s3_client()

    try:
        s3_client.upload_file(str(file_path), S3_BUCKET_RAW, s3_key)
        logger.info(f"Uploaded: s3://{S3_BUCKET_RAW}/{s3_key}")
        return s3_key
    except Exception as e:
        logger.error(f"S3 upload failed for {s3_key}: {e}")
        raise


def _fitbit_get(access_token: str, url: str) -> Dict:
    """Make GET request to Fitbit API."""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Fitbit API error: {url} - {e}")
        raise


def fetch_steps_intraday(access_token: str, date_str: str, detail_level: str = "1min") -> pd.DataFrame:
    """Fetch intraday steps data."""
    url = f"https://api.fitbit.com/1/user/-/activities/steps/date/{date_str}/1d/{detail_level}.json"
    data = _fitbit_get(access_token, url)

    intraday = data.get("activities-steps-intraday", {}).get("dataset", [])
    if not intraday:
        return pd.DataFrame()

    df = pd.DataFrame(intraday)
    df["datetime"] = pd.to_datetime(f"{date_str} " + df["time"], format="%Y-%m-%d %H:%M:%S")
    return df[["datetime", "time", "value"]]


def fetch_calories_intraday(access_token: str, date_str: str, detail_level: str = "1min") -> pd.DataFrame:
    """Fetch intraday calories data."""
    url = f"https://api.fitbit.com/1/user/-/activities/calories/date/{date_str}/1d/{detail_level}.json"
    data = _fitbit_get(access_token, url)

    intraday = data.get("activities-calories-intraday", {}).get("dataset", [])
    if not intraday:
        return pd.DataFrame()

    df = pd.DataFrame(intraday)
    df["datetime"] = pd.to_datetime(f"{date_str} " + df["time"], format="%Y-%m-%d %H:%M:%S")
    return df[["datetime", "time", "value"]]


def fetch_azm_intraday(access_token: str, date_str: str, detail_level: str = "1min") -> pd.DataFrame:
    """Fetch intraday active zone minutes data."""
    url = f"https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{date_str}/1d/{detail_level}.json"
    data = _fitbit_get(access_token, url)

    intraday = data.get("activities-active-zone-minutes-intraday", {}).get("dataset", [])
    if not intraday:
        return pd.DataFrame()

    df = pd.DataFrame(intraday)
    df["datetime"] = pd.to_datetime(f"{date_str} " + df["time"], format="%Y-%m-%d %H:%M:%S")
    return df[["datetime", "time", "value"]]


def fetch_hrv_daily(access_token: str, date_str: str) -> pd.DataFrame:
    """Fetch daily HRV data."""
    url = f"https://api.fitbit.com/1/user/-/hrv/date/{date_str}/all.json"
    data = _fitbit_get(access_token, url)

    items = data.get("hrv", [])
    if not items:
        return pd.DataFrame()

    return pd.DataFrame(items)


def fetch_wrist_temp_daily(access_token: str, date_str: str) -> pd.DataFrame:
    """Fetch daily wrist temperature data."""
    url = f"https://api.fitbit.com/1/user/-/temp/skin/date/{date_str}/1d.json"
    data = _fitbit_get(access_token, url)

    items = data.get("tempSkin", [])
    if not items:
        return pd.DataFrame()

    return pd.DataFrame(items)


def main():
    """Run pipeline for all users."""
    date_str = date.today().strftime("%Y-%m-%d")
    logger.info(f"Starting pipeline for {date_str}")

    for user_id in USER_IDS:
        logger.info(f"\nProcessing user: {user_id}")

        try:
            access_token = get_access_token(user_id)
        except Exception as e:
            logger.error(f"Failed to get access token for {user_id}: {e}")
            continue

        # Fetch and upload metrics
        metrics = [
            (fetch_steps_intraday, "steps", access_token, date_str),
            (fetch_calories_intraday, "calories", access_token, date_str),
            (fetch_azm_intraday, "azm", access_token, date_str),
            (fetch_hrv_daily, "hrv", access_token, date_str),
            (fetch_wrist_temp_daily, "wrist_temp", access_token, date_str),
        ]

        for fetch_func, metric_name, token, dt_str in metrics:
            try:
                logger.info(f"  Fetching {metric_name}...")
                df = fetch_func(token, dt_str)

                if df.empty:
                    logger.warning(f"  No data for {metric_name}")
                    continue

                csv_path = save_df_to_csv(df, user_id, metric_name)
                upload_file_to_s3(csv_path, user_id, metric_name)
                logger.info(f"  ✅ {metric_name} complete")

            except Exception as e:
                logger.error(f"  ❌ {metric_name} failed: {e}")

    logger.info("\nPipeline complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
