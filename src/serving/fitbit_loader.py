"""
Load and normalize Fitbit data from S3.

Handles multiple JSON files with different structures.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

import boto3


class FitbitLoader:
    """Loads Fitbit data from S3 bucket."""
    
    def __init__(self):
        self.bucket = os.environ.get("FITBIT_BUCKET")
        self.prefix = os.environ.get("FITBIT_PREFIX", "fitbit_data/")
        self.s3 = boto3.client("s3")
    
    def load_user_data(self, user_id: str, date: str = None) -> Dict:
        """
        Load all available Fitbit data for a user/date.
        
        Returns dict with normalized data:
        {
            "heart_rate": [{"timestamp": ..., "value": ...}, ...],
            "steps": [...],
            "calories": [...],
            "sleep": [...],
            ...
        }
        """
        if not date:
            date = self._get_latest_date(user_id)
            if not date:
                return None
        
        prefix = f"{self.prefix}{user_id}/{date}/"
        
        # List all files for this user/date
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        
        if "Contents" not in resp:
            return None
        
        data = {"_meta": {"user_id": user_id, "date": date, "files": []}}
        
        for obj in resp["Contents"]:
            key = obj["Key"]
            filename = key.split("/")[-1]
            
            if not filename.endswith(".json"):
                continue
            
            try:
                raw = self._load_json(key)
                parsed = self._parse_file(filename, raw)
                
                for data_type, records in parsed.items():
                    if data_type not in data:
                        data[data_type] = []
                    data[data_type].extend(records)
                
                data["_meta"]["files"].append(filename)
            except Exception as e:
                print(f"Error parsing {key}: {e}")
        
        return data if len(data) > 1 else None
    
    def _load_json(self, key: str) -> dict:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    
    def _get_latest_date(self, user_id: str) -> Optional[str]:
        prefix = f"{self.prefix}{user_id}/"
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter="/")
        
        dates = []
        for p in resp.get("CommonPrefixes", []):
            date_str = p["Prefix"].replace(prefix, "").strip("/")
            dates.append(date_str)
        
        return max(dates) if dates else None
    
    def _parse_file(self, filename: str, raw: dict) -> Dict[str, List]:
        """
        Parse different Fitbit JSON formats into normalized structure.
        
        Returns: {"data_type": [{"timestamp": ..., "value": ...}, ...]}
        """
        name = filename.replace(".json", "").lower()
        
        # Heart rate intraday
        if "heart" in name and "intraday" in name:
            return self._parse_heart_intraday(raw)
        
        # Heart daily summary
        if "heart" in name and "daily" in name:
            return self._parse_heart_daily(raw)
        
        # Calories/Steps/Distance/Floors intraday
        if "intraday" in name:
            return self._parse_intraday(name, raw)
        
        # Activities summary
        if "activities" in name:
            return self._parse_activities(raw)
        
        # Sleep
        if "sleep" in name:
            return self._parse_sleep(raw)
        
        # Generic fallback
        return self._parse_generic(name, raw)
    
    def _parse_heart_intraday(self, raw: dict) -> Dict[str, List]:
        """Parse heart rate intraday data."""
        records = []
        
        # Fitbit format: activities-heart-intraday
        intraday = raw.get("activities-heart-intraday", {}).get("dataset", [])
        date = raw.get("activities-heart", [{}])[0].get("dateTime", "")
        
        for item in intraday:
            records.append({
                "timestamp": f"{date}T{item.get('time', '')}",
                "value": item.get("value")
            })
        
        return {"heart_rate": records}
    
    def _parse_heart_daily(self, raw: dict) -> Dict[str, List]:
        """Parse heart daily summary (resting HR, zones)."""
        records = []
        
        for day in raw.get("activities-heart", []):
            date = day.get("dateTime", "")
            value = day.get("value", {})
            
            records.append({
                "timestamp": date,
                "resting_hr": value.get("restingHeartRate"),
                "zones": value.get("heartRateZones", [])
            })
        
        return {"heart_daily": records}
    
    def _parse_intraday(self, name: str, raw: dict) -> Dict[str, List]:
        """Parse generic intraday data (calories, steps, etc.)."""
        # Determine data type
        if "calories" in name:
            key = "activities-calories-intraday"
            data_type = "calories"
        elif "steps" in name:
            key = "activities-steps-intraday"
            data_type = "steps"
        elif "distance" in name:
            key = "activities-distance-intraday"
            data_type = "distance"
        elif "floors" in name:
            key = "activities-floors-intraday"
            data_type = "floors"
        elif "elevation" in name:
            key = "activities-elevation-intraday"
            data_type = "elevation"
        else:
            return {}
        
        records = []
        intraday = raw.get(key, {}).get("dataset", [])
        
        # Get date from summary
        summary_key = key.replace("-intraday", "")
        date = raw.get(summary_key, [{}])[0].get("dateTime", "")
        
        for item in intraday:
            records.append({
                "timestamp": f"{date}T{item.get('time', '')}",
                "value": item.get("value")
            })
        
        return {data_type: records}
    
    def _parse_activities(self, raw: dict) -> Dict[str, List]:
        """Parse activities summary."""
        records = []
        
        # Daily summary
        summary = raw.get("summary", {})
        if summary:
            records.append({
                "steps": summary.get("steps"),
                "calories": summary.get("caloriesOut"),
                "distance": summary.get("distances", [{}])[0].get("distance") if summary.get("distances") else None,
                "active_minutes": summary.get("fairlyActiveMinutes", 0) + summary.get("veryActiveMinutes", 0),
                "sedentary_minutes": summary.get("sedentaryMinutes")
            })
        
        return {"activities_summary": records}
    
    def _parse_sleep(self, raw: dict) -> Dict[str, List]:
        """Parse sleep data."""
        records = []
        
        for sleep in raw.get("sleep", []):
            records.append({
                "start_time": sleep.get("startTime"),
                "end_time": sleep.get("endTime"),
                "duration_ms": sleep.get("duration"),
                "efficiency": sleep.get("efficiency"),
                "minutes_asleep": sleep.get("minutesAsleep"),
                "minutes_awake": sleep.get("minutesAwake"),
                "stages": sleep.get("levels", {}).get("summary", {})
            })
        
        return {"sleep": records}
    
    def _parse_generic(self, name: str, raw: dict) -> Dict[str, List]:
        """Fallback parser for unknown formats."""
        if isinstance(raw, list):
            return {name: raw}
        elif isinstance(raw, dict):
            return {name: [raw]}
        return {}
    
    def list_users(self) -> List[str]:
        """List all users in bucket."""
        users = set()
        paginator = self.s3.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter="/"):
            for p in page.get("CommonPrefixes", []):
                user_id = p["Prefix"].replace(self.prefix, "").strip("/")
                if user_id:
                    users.add(user_id)
        
        return list(users)