"""
Data Preprocessing Pipeline for CBT Prediction Platform

This script prepares raw data for the training pipeline by:
1. Reading raw data files from categorized folders
2. Normalizing column names and formats
3. Converting units (F to C for environmental, Central to UTC for CBT)
4. Outputting files in the format expected by prepare_data.py

Input Structure (raw_data/):
    Final Merge _ CBT Files/
        user1_cbt.csv, user2_cbt.csv, ...
    Final Merge _ Heart Rate Files/
        user1_heart_rate.csv, user2_heart_rate.csv, ...
    Final Merge _ Wrist Temperature Files/
        user1_wrist_temp.csv, user2_wrist_temp.csv, ...
    Final Merge_ Environmental Files/
        user1_environmental.csv, user2_environmental.csv, ...

Output Structure (preprocessed_data/):
    user1/
        heart_rate.csv
        skin_temperature.csv
        environmental.csv
        cbt_labels.csv
    user2/
        ...
    all_users/
        heart_rate.csv          (combined, with user_id column)
        skin_temperature.csv
        environmental.csv
        cbt_labels.csv

The output format matches what prepare_data.py and DataLoader expect.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for the preprocessing pipeline."""
    
    SCRIPT_DIR = Path(__file__).parent.resolve()
    RAW_DATA_DIR = SCRIPT_DIR / "raw_data"
    OUTPUT_DIR = SCRIPT_DIR / "preprocessed_data"
    
    # Folder names in raw_data (case-insensitive matching)
    CBT_FOLDER_PATTERN = "cbt"
    HR_FOLDER_PATTERN = "heart rate"
    WRIST_TEMP_FOLDER_PATTERN = "wrist temp"
    ENV_FOLDER_PATTERN = "environmental"
    
    # Output filenames (must match what DataLoader expects)
    HR_OUTPUT = "heart_rate.csv"
    SKIN_TEMP_OUTPUT = "skin_temperature.csv"
    ENV_OUTPUT = "environmental.csv"
    CBT_OUTPUT = "cbt_labels.csv"
    
    # Timezone for CBT data (manual measurements in local time)
    LOCAL_TIMEZONE = "America/Chicago"


# ============================================
# UTILITIES
# ============================================

def extract_user_id(filename: str) -> str:
    """
    Extract user ID from filename.
    
    Expected format: user1_something.csv, user2_data.csv, etc.
    
    Args:
        filename: Name of the file
    
    Returns:
        User ID (e.g., "user1", "user2")
    """
    filename_lower = filename.lower()
    
    # Pattern: user followed by number at start of filename
    match = re.match(r'^(user\d+)', filename_lower)
    if match:
        return match.group(1)
    
    # Fallback: first part before underscore
    parts = Path(filename).stem.split('_')
    if parts:
        return parts[0].lower()
    
    return "unknown"


def find_folder(base_dir: Path, pattern: str) -> Optional[Path]:
    """
    Find a folder containing the pattern in its name.
    
    Args:
        base_dir: Directory to search in
        pattern: Pattern to match (case-insensitive)
    
    Returns:
        Path to matching folder, or None
    """
    pattern_lower = pattern.lower()
    
    for item in base_dir.iterdir():
        if item.is_dir() and pattern_lower in item.name.lower():
            return item
    
    return None


def load_csv_flexible(filepath: Path) -> pd.DataFrame:
    """
    Load CSV with flexible parsing.
    
    Handles various delimiters and encodings.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame
    """
    try:
        # Try standard CSV first
        df = pd.read_csv(filepath)
        if len(df.columns) > 1:
            return df
    except:
        pass
    
    try:
        # Try with different encoding
        df = pd.read_csv(filepath, encoding='latin-1')
        if len(df.columns) > 1:
            return df
    except:
        pass
    
    try:
        # Try tab-separated
        df = pd.read_csv(filepath, sep='\t')
        if len(df.columns) > 1:
            return df
    except:
        pass
    
    # Last resort
    return pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')


# ============================================
# HEART RATE PROCESSOR
# ============================================

class HeartRateProcessor:
    """
    Process Fitbit heart rate files.
    
    Input format (from Fitbit):
        timestamp, beats per minute
        2025-04-10T00:00:01Z, 67
    
    Output format (for DataLoader.load_fitbit_hr):
        timestamp, beats per minute
        2025-04-10T00:00:01Z, 67
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}  # user_id -> DataFrame
    
    def process_folder(self, folder: Path) -> "HeartRateProcessor":
        """Process all heart rate files in folder."""
        if folder is None or not folder.exists():
            print("  No heart rate folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} heart rate files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing HR", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        # Sort and deduplicate per user
        for user_id in self.data:
            self.data[user_id] = (
                self.data[user_id]
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="first")
                .reset_index(drop=True)
            )
            print(f"    {user_id}: {len(self.data[user_id]):,} samples")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single heart rate file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Find timestamp column
        ts_col = None
        for c in df.columns:
            if "timestamp" in c or "time" in c:
                ts_col = c
                break
        
        # Find heart rate column
        hr_col = None
        for c in df.columns:
            if "beat" in c or "bpm" in c or "heart" in c:
                hr_col = c
                break
        
        if ts_col is None or hr_col is None:
            print(f"    Warning: Missing columns in {filepath.name}")
            print(f"    Available: {list(df.columns)}")
            return None
        
        # Create output DataFrame with expected column names
        result = pd.DataFrame()
        result["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        result["beats per minute"] = pd.to_numeric(df[hr_col], errors="coerce")
        
        # Remove invalid rows
        result = result.dropna(subset=["timestamp", "beats per minute"])
        result = result[(result["beats per minute"] >= 30) & (result["beats per minute"] <= 220)]
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """
        Save processed heart rate data.
        
        Args:
            output_dir: Base output directory
            per_user: Save individual user files
            combined: Save combined file with user_id column
        
        Returns:
            Dict of saved file paths
        """
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.HR_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_hr"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df = combined_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            
            filepath = all_users_dir / Config.HR_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_hr"] = filepath
        
        return saved


# ============================================
# WRIST TEMPERATURE PROCESSOR
# ============================================

class WristTemperatureProcessor:
    """
    Process Fitbit wrist/skin temperature files.
    
    Input format (from Fitbit):
        recorded_time, temperature
        2025-05-30T00:00, -2.095291443
    
    Output format (for DataLoader.load_fitbit_skin_temp):
        recorded_time, temperature
        2025-05-30T00:00, -2.095291443
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
    
    def process_folder(self, folder: Path) -> "WristTemperatureProcessor":
        """Process all wrist temperature files in folder."""
        if folder is None or not folder.exists():
            print("  No wrist temperature folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} wrist temperature files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing Skin Temp", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        # Sort and deduplicate per user
        for user_id in self.data:
            self.data[user_id] = (
                self.data[user_id]
                .sort_values("recorded_time")
                .drop_duplicates(subset=["recorded_time"], keep="first")
                .reset_index(drop=True)
            )
            print(f"    {user_id}: {len(self.data[user_id]):,} samples")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single wrist temperature file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Find timestamp column
        ts_col = None
        for c in df.columns:
            if "recorded" in c:
                ts_col = c
                break
        if ts_col is None:
            for c in df.columns:
                if "timestamp" in c or "time" in c:
                    ts_col = c
                    break
        
        # Find temperature column
        temp_col = None
        for c in df.columns:
            if "temp" in c:
                temp_col = c
                break
        
        if ts_col is None or temp_col is None:
            print(f"    Warning: Missing columns in {filepath.name}")
            print(f"    Available: {list(df.columns)}")
            return None
        
        # Create output DataFrame with expected column names
        result = pd.DataFrame()
        result["recorded_time"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        result["temperature"] = pd.to_numeric(df[temp_col], errors="coerce")
        
        result = result.dropna(subset=["recorded_time"])
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """Save processed wrist temperature data."""
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.SKIN_TEMP_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_skin"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df = combined_df.sort_values(["user_id", "recorded_time"]).reset_index(drop=True)
            
            filepath = all_users_dir / Config.SKIN_TEMP_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_skin"] = filepath
        
        return saved


# ============================================
# ENVIRONMENTAL PROCESSOR
# ============================================

class EnvironmentalProcessor:
    """
    Process Govee environmental monitor files.
    
    Input format (from Govee):
        Timestamp for sample frequency every 1 min min, PM2.5(µg/m³), Temperature_Fahrenheit, Relative_Humidity
        2025-04-07 16:51:00, 0, 69.98, 50.6
    
    Output format (for DataLoader.load_govee_env):
        Timestamp for sample frequency every 1 min min, Temperature_Fahrenheit, Relative_Humidity
        2025-04-07 16:51:00, 69.98, 50.6
    
    Note: Keep temperature in Fahrenheit - DataLoader will convert to Celsius.
    Note: PM2.5 is dropped as it's not used.
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
    
    def process_folder(self, folder: Path) -> "EnvironmentalProcessor":
        """Process all environmental files in folder."""
        if folder is None or not folder.exists():
            print("  No environmental folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} environmental files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing Env", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        # Sort and deduplicate per user
        for user_id in self.data:
            self.data[user_id] = (
                self.data[user_id]
                .sort_values("Timestamp for sample frequency every 1 min min")
                .drop_duplicates(subset=["Timestamp for sample frequency every 1 min min"], keep="first")
                .reset_index(drop=True)
            )
            print(f"    {user_id}: {len(self.data[user_id]):,} samples")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single environmental file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names for detection (but we'll rename to expected format)
        original_columns = df.columns.tolist()
        df.columns = (
            df.columns
            .str.normalize("NFKC")
            .str.replace("\u00A0", " ", regex=False)
            .str.lower()
            .str.strip()
        )
        
        # Find columns
        ts_col = None
        for i, c in enumerate(df.columns):
            if c.startswith("timestamp") or c == "time" or c == "datetime":
                ts_col = i
                break
        
        temp_col = None
        for i, c in enumerate(df.columns):
            if "temperature" in c and "pm" not in c:
                temp_col = i
                break
        
        humid_col = None
        for i, c in enumerate(df.columns):
            if "humid" in c:
                humid_col = i
                break
        
        if ts_col is None:
            print(f"    Warning: No timestamp column in {filepath.name}")
            print(f"    Available: {original_columns}")
            return None
        
        if temp_col is None and humid_col is None:
            print(f"    Warning: No temp/humidity in {filepath.name}")
            return None
        
        # Build output with expected column names
        result = pd.DataFrame()
        
        # Parse timestamp (keep as string in expected format)
        ts_values = pd.to_datetime(df.iloc[:, ts_col], errors="coerce")
        result["Timestamp for sample frequency every 1 min min"] = ts_values.dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Temperature (keep in Fahrenheit)
        if temp_col is not None:
            result["Temperature_Fahrenheit"] = pd.to_numeric(df.iloc[:, temp_col], errors="coerce")
        
        # Humidity
        if humid_col is not None:
            result["Relative_Humidity"] = pd.to_numeric(df.iloc[:, humid_col], errors="coerce")
        
        result = result.dropna(subset=["Timestamp for sample frequency every 1 min min"])
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """Save processed environmental data."""
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.ENV_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_env"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            combined_df = combined_df.sort_values(
                ["user_id", "Timestamp for sample frequency every 1 min min"]
            ).reset_index(drop=True)
            
            filepath = all_users_dir / Config.ENV_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_env"] = filepath
        
        return saved


# ============================================
# CBT PROCESSOR
# ============================================

class CBTProcessor:
    """
    Process CBT (Core Body Temperature) label files.
    
    Input format (manual measurements):
        Date:, Time:, CBT (Deg F):
        4/5/2025, 8:59 PM, 98.5
    
    Output format (for DataLoader.load_cbt):
        Date:, Time:, CBT (Deg F):
        4/5/2025, 8:59 PM, 98.5
    
    Note: Keep in original format - DataLoader handles timezone/unit conversion.
    """
    
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
    
    def process_folder(self, folder: Path) -> "CBTProcessor":
        """Process all CBT files in folder."""
        if folder is None or not folder.exists():
            print("  No CBT folder found")
            return self
        
        files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        print(f"  Found {len(files)} CBT files in {folder.name}")
        
        for filepath in tqdm(files, desc="  Processing CBT", disable=not HAS_TQDM):
            user_id = extract_user_id(filepath.name)
            df = self._process_file(filepath)
            
            if df is not None and len(df) > 0:
                if user_id in self.data:
                    self.data[user_id] = pd.concat([self.data[user_id], df], ignore_index=True)
                else:
                    self.data[user_id] = df
        
        for user_id in self.data:
            print(f"    {user_id}: {len(self.data[user_id])} measurements")
        
        return self
    
    def _process_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Process a single CBT file."""
        try:
            df = load_csv_flexible(filepath)
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")
            return None
        
        if df.empty:
            return None
        
        # Normalize column names for detection
        df.columns = df.columns.str.strip()
        
        # Find columns by pattern (handle colons in names)
        date_col = None
        time_col = None
        cbt_col = None
        
        for c in df.columns:
            c_lower = c.lower().rstrip(':')
            if c_lower == "date":
                date_col = c
            elif c_lower == "time":
                time_col = c
            elif "cbt" in c_lower or "core" in c_lower or "body" in c_lower:
                cbt_col = c
        
        if date_col is None or time_col is None or cbt_col is None:
            print(f"    Warning: Missing columns in {filepath.name}")
            print(f"    Available: {list(df.columns)}")
            print(f"    Found: date={date_col}, time={time_col}, cbt={cbt_col}")
            return None
        
        # Build output with expected column names
        result = pd.DataFrame()
        result["Date:"] = df[date_col].astype(str).str.strip()
        result["Time:"] = df[time_col].astype(str).str.strip()
        result["CBT (Deg F):"] = pd.to_numeric(df[cbt_col], errors="coerce")
        
        # Validate CBT values are in reasonable range
        result = result[
            (result["CBT (Deg F):"] >= 95) & 
            (result["CBT (Deg F):"] <= 105)
        ]
        
        return result
    
    def save(self, output_dir: Path, per_user: bool = True, combined: bool = True) -> Dict[str, Path]:
        """Save processed CBT data."""
        saved = {}
        
        if per_user:
            for user_id, df in self.data.items():
                user_dir = output_dir / user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = user_dir / Config.CBT_OUTPUT
                df.to_csv(filepath, index=False)
                saved[f"{user_id}_cbt"] = filepath
        
        if combined and self.data:
            all_users_dir = output_dir / "all_users"
            all_users_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df = []
            for user_id, df in self.data.items():
                df_copy = df.copy()
                df_copy["user_id"] = user_id
                combined_df.append(df_copy)
            
            combined_df = pd.concat(combined_df, ignore_index=True)
            
            filepath = all_users_dir / Config.CBT_OUTPUT
            combined_df.to_csv(filepath, index=False)
            saved["combined_cbt"] = filepath
        
        return saved


# ============================================
# MAIN PIPELINE
# ============================================

def run_pipeline(
    raw_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    per_user: bool = True,
    combined: bool = True
) -> Dict[str, Path]:
    """
    Run the full preprocessing pipeline.
    
    Args:
        raw_data_dir: Directory containing raw data folders
        output_dir: Directory for preprocessed output
        per_user: Save individual user files
        combined: Save combined files with user_id column
    
    Returns:
        Dict of saved file paths
    """
    raw_data_dir = raw_data_dir or Config.RAW_DATA_DIR
    output_dir = output_dir or Config.OUTPUT_DIR
    
    print("=" * 60)
    print("CBT Prediction Platform - Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nRaw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Per-user output: {per_user}")
    print(f"Combined output: {combined}")
    print()
    
    # Validate raw data directory
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    # Find data folders
    print("Step 1: Discovering data folders...")
    hr_folder = find_folder(raw_data_dir, Config.HR_FOLDER_PATTERN)
    wrist_folder = find_folder(raw_data_dir, Config.WRIST_TEMP_FOLDER_PATTERN)
    env_folder = find_folder(raw_data_dir, Config.ENV_FOLDER_PATTERN)
    cbt_folder = find_folder(raw_data_dir, Config.CBT_FOLDER_PATTERN)
    
    print(f"  Heart Rate folder: {hr_folder.name if hr_folder else 'NOT FOUND'}")
    print(f"  Wrist Temp folder: {wrist_folder.name if wrist_folder else 'NOT FOUND'}")
    print(f"  Environmental folder: {env_folder.name if env_folder else 'NOT FOUND'}")
    print(f"  CBT folder: {cbt_folder.name if cbt_folder else 'NOT FOUND'}")
    print()
    
    all_saved = {}
    
    # Process Heart Rate
    print("Step 2: Processing heart rate data...")
    hr_processor = HeartRateProcessor().process_folder(hr_folder)
    hr_saved = hr_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(hr_saved)
    print()
    
    # Process Wrist Temperature
    print("Step 3: Processing wrist temperature data...")
    wrist_processor = WristTemperatureProcessor().process_folder(wrist_folder)
    wrist_saved = wrist_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(wrist_saved)
    print()
    
    # Process Environmental
    print("Step 4: Processing environmental data...")
    env_processor = EnvironmentalProcessor().process_folder(env_folder)
    env_saved = env_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(env_saved)
    print()
    
    # Process CBT Labels
    print("Step 5: Processing CBT label data...")
    cbt_processor = CBTProcessor().process_folder(cbt_folder)
    cbt_saved = cbt_processor.save(output_dir, per_user=per_user, combined=combined)
    all_saved.update(cbt_saved)
    print()
    
    # Summary
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    print(f"\nOutput directory: {output_dir}")
    
    # List users processed
    all_users = set()
    for processor in [hr_processor, wrist_processor, env_processor, cbt_processor]:
        all_users.update(processor.data.keys())
    
    print(f"\nUsers processed: {len(all_users)}")
    for user_id in sorted(all_users):
        hr_count = len(hr_processor.data.get(user_id, []))
        skin_count = len(wrist_processor.data.get(user_id, []))
        env_count = len(env_processor.data.get(user_id, []))
        cbt_count = len(cbt_processor.data.get(user_id, []))
        print(f"  {user_id}: HR={hr_count:,}, Skin={skin_count:,}, Env={env_count:,}, CBT={cbt_count}")
    
    if combined:
        print(f"\nCombined files saved to: {output_dir / 'all_users'}")
    
    if per_user:
        print(f"\nPer-user files saved to: {output_dir}/<user_id>/")
    
    print("\nFiles are ready for prepare_data.py")
    print("Usage:")
    print(f"  python -m src.training.prepare_data --data-dir {output_dir / 'all_users'}")
    print("  OR for single user:")
    print(f"  python -m src.training.prepare_data --data-dir {output_dir}/<user_id>")
    
    return all_saved


def validate_output(output_dir: Path) -> bool:
    """
    Validate that output files match what prepare_data.py expects.
    
    Args:
        output_dir: Directory to validate
    
    Returns:
        True if all expected files exist
    """
    all_users_dir = output_dir / "all_users"
    
    expected_files = [
        Config.HR_OUTPUT,
        Config.SKIN_TEMP_OUTPUT,
        Config.ENV_OUTPUT,
        Config.CBT_OUTPUT
    ]
    
    print("\nValidating output files...")
    all_ok = True
    
    for filename in expected_files:
        filepath = all_users_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"  ✓ {filename}: {len(df):,} rows")
        else:
            print(f"  ✗ {filename}: NOT FOUND")
            all_ok = False
    
    return all_ok


# ============================================
# CLI
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess raw data for CBT prediction training"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw data directory (default: preprocessing/raw_data)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: preprocessing/preprocessed_data)"
    )
    parser.add_argument(
        "--no-per-user",
        action="store_true",
        help="Don't save individual user files"
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Don't save combined files"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing output files"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        output_dir = args.output_dir or Config.OUTPUT_DIR
        validate_output(output_dir)
    else:
        saved = run_pipeline(
            raw_data_dir=args.raw_dir,
            output_dir=args.output_dir,
            per_user=not args.no_per_user,
            combined=not args.no_combined
        )
        
        # Validate output
        output_dir = args.output_dir or Config.OUTPUT_DIR
        validate_output(output_dir)

