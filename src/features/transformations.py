"""
Feature Transformations for CBT Prediction

CRITICAL: This file is used by BOTH training AND serving.
Any changes here must be reflected in both pipelines.

DO NOT modify feature calculations without retraining the model.

Data Sources:
- Fitbit: Heart Rate (6-11 samples/min), Skin/Wrist Temperature (~1/min)
- Govee: Ambient Temperature, Humidity (~1/min)

Target (training only): Core Body Temperature from Braun thermometer

Feature Naming Convention:
- Skin Temperature: temp_*
- Heart Rate: bpm_*
- Ambient Temperature: temp_env_*
- Humidity: humidity_env_*

Total Features: 86 (84 training + 2 metadata)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy import stats


# ============================================
# FEATURE CONFIGURATION
# ============================================

# Rolling window sizes (in minutes)
ROLLING_WINDOWS = [5, 20, 35]

# Lag interval for subsampled rolling stats (every 10th minute)
LAG_INTERVAL_MINUTES = 10

# Minimum samples required for valid prediction (ALL signals required)
MIN_SAMPLES = {
    "heart_rate": 30,       # ~5 min at 6 samples/min
    "skin_temperature": 5,  # ~5 min at 1 sample/min
    "ambient_temp": 5,      # ~5 min at 1 sample/min
    "humidity": 5           # ~5 min at 1 sample/min
}


class FeatureTransformer:
    """
    Transforms raw Fitbit and environmental data into ML features.
    
    Used by:
        - Training pipeline (prepare_data.py)
        - Serving pipeline (handler.py)
    
    Features computed (86 total):
        - Skin Temperature (21): current, rolling stats, lagged stats, diff, slope
        - Heart Rate (21): current, rolling stats, lagged stats, diff, slope
        - Ambient Temperature (21): current, rolling stats, lagged stats, diff, slope
        - Humidity (21): current, rolling stats, lagged stats, diff, slope
        - Metadata (2): user_id, timestamp
    
    REQUIREMENT: All four signals (skin temp, heart rate, ambient temp, humidity)
    are REQUIRED for valid prediction.
    """
    
    def __init__(self, rolling_windows: List[int] = None):
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self._feature_names: List[str] = []
    
    def transform(
        self,
        fitbit_df: pd.DataFrame,
        env_df: pd.DataFrame,
        target_timestamp: Optional[pd.Timestamp] = None,
        user_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Transform raw data into features for a single prediction point.
        
        Args:
            fitbit_df: Fitbit data with columns:
                - timestamp: datetime (UTC)
                - heart_rate: int (beats per minute)
                - skin_temperature: float (deviation from baseline)
            env_df: Environmental data with columns:
                - timestamp: datetime (UTC)
                - ambient_temp: float (Celsius)
                - humidity: float (relative humidity %)
            target_timestamp: Compute features up to this point only (backward-looking).
            user_id: User identifier for the data.
        
        Returns:
            DataFrame with single row of 86 features, or None if insufficient data.
        """
        # Validate and prepare Fitbit data
        fitbit_df = self._prepare_fitbit(fitbit_df)
        if fitbit_df is None:
            return None
        
        # Validate and prepare environmental data (REQUIRED)
        env_df = self._prepare_env(env_df)
        if env_df is None:
            return None
        
        # Determine reference timestamp
        if target_timestamp is not None:
            ref_time = pd.to_datetime(target_timestamp)
            if ref_time.tz is None:
                ref_time = ref_time.tz_localize("UTC")
            # Filter to only data before target (backward-looking only)
            fitbit_df = fitbit_df[fitbit_df["timestamp"] <= ref_time].copy()
            env_df = env_df[env_df["timestamp"] <= ref_time].copy()
        else:
            ref_time = fitbit_df["timestamp"].iloc[-1]
        
        # Validate minimum data requirements for ALL signals
        validation = self._validate_data(fitbit_df, env_df)
        if not validation["valid"]:
            print(f"Insufficient data: {validation['missing']}")
            return None
        
        # Build features
        features = {}
        
        # 1. Skin Temperature features (21 features, prefix: temp_)
        temp_features = self._compute_signal_features(
            df=fitbit_df,
            col="skin_temperature",
            prefix="temp",
            current_name="temperature",
            ref_time=ref_time
        )
        features.update(temp_features)
        
        # 2. Heart Rate features (21 features, prefix: bpm_)
        bpm_features = self._compute_signal_features(
            df=fitbit_df,
            col="heart_rate",
            prefix="bpm",
            current_name="bpm",
            ref_time=ref_time
        )
        features.update(bpm_features)
        
        # 3. Ambient Temperature features (21 features, prefix: temp_env_)
        temp_env_features = self._compute_signal_features(
            df=env_df,
            col="ambient_temp",
            prefix="temp_env",
            current_name="env_Temperature_Celsius",
            ref_time=ref_time
        )
        features.update(temp_env_features)
        
        # 4. Humidity features (21 features, prefix: humidity_env_)
        humidity_features = self._compute_signal_features(
            df=env_df,
            col="humidity",
            prefix="humidity_env",
            current_name="Relative_Humidity",
            ref_time=ref_time
        )
        features.update(humidity_features)
        
        # 5. Metadata (2 features, not used in training)
        features["user_id"] = user_id if user_id else "unknown"
        features["timestamp"] = ref_time.isoformat()
        
        # Reorder columns to match expected order
        result = pd.DataFrame([features])
        result = self._reorder_columns(result)
        
        self._feature_names = list(result.columns)
        
        return result
    
    def _prepare_fitbit(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare Fitbit data: parse timestamps, sort, validate."""
        if df is None or len(df) == 0:
            return None
        
        df = df.copy()
        
        # Handle column name variations
        col_mapping = {
            "beats per minute": "heart_rate",
            "bpm": "heart_rate",
            "heartrate": "heart_rate",
            "recorded_time": "timestamp",
            "time": "timestamp",
            "temperature": "skin_temperature",
            "wrist_temperature": "skin_temperature",
            "wrist_temp": "skin_temperature",
        }
        
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns=col_mapping)
        
        # Parse timestamp
        if "timestamp" not in df.columns:
            raise ValueError("Fitbit data must have 'timestamp' column")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Validate required columns exist
        if "heart_rate" not in df.columns:
            raise ValueError("Fitbit data must have 'heart_rate' column")
        if "skin_temperature" not in df.columns:
            raise ValueError("Fitbit data must have 'skin_temperature' column")
        
        # Convert to numeric
        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
        df["skin_temperature"] = pd.to_numeric(df["skin_temperature"], errors="coerce")
        
        return df
    
    def _prepare_env(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare environmental data: parse, convert units, clean."""
        if df is None or len(df) == 0:
            return None
        
        df = df.copy()
        
        # Handle column name variations
        col_mapping = {
            "timestamp for sample frequency every 1 min min": "timestamp",
            "time": "timestamp",
            "temperature_fahrenheit": "ambient_temp",
            "temperature_celsius": "ambient_temp",
            "env_temperature_celsius": "ambient_temp",
            "temperature": "ambient_temp",
            "temp": "ambient_temp",
            "relative_humidity": "humidity",
            "rh": "humidity",
        }
        
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns=col_mapping)
        
        # Drop PM2.5 if present
        pm_cols = [c for c in df.columns if "pm2.5" in c.lower()]
        df = df.drop(columns=pm_cols, errors="ignore")
        
        # Parse timestamp
        if "timestamp" not in df.columns:
            raise ValueError("Environmental data must have 'timestamp' column")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Validate required columns
        if "ambient_temp" not in df.columns:
            raise ValueError("Environmental data must have temperature column")
        if "humidity" not in df.columns:
            raise ValueError("Environmental data must have humidity column")
        
        # Convert to numeric
        df["ambient_temp"] = pd.to_numeric(df["ambient_temp"], errors="coerce")
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        
        # Convert temperature if in Fahrenheit (values > 50 assumed Fahrenheit)
        if df["ambient_temp"].median() > 50:
            df["ambient_temp"] = (df["ambient_temp"] - 32) * 5 / 9
        
        return df
    
    def _validate_data(self, fitbit_df: pd.DataFrame, env_df: pd.DataFrame) -> Dict:
        """Validate that all required signals have minimum samples."""
        missing = []
        
        hr_count = fitbit_df["heart_rate"].notna().sum()
        if hr_count < MIN_SAMPLES["heart_rate"]:
            missing.append(f"heart_rate ({hr_count}/{MIN_SAMPLES['heart_rate']})")
        
        skin_count = fitbit_df["skin_temperature"].notna().sum()
        if skin_count < MIN_SAMPLES["skin_temperature"]:
            missing.append(f"skin_temperature ({skin_count}/{MIN_SAMPLES['skin_temperature']})")
        
        ambient_count = env_df["ambient_temp"].notna().sum()
        if ambient_count < MIN_SAMPLES["ambient_temp"]:
            missing.append(f"ambient_temp ({ambient_count}/{MIN_SAMPLES['ambient_temp']})")
        
        humidity_count = env_df["humidity"].notna().sum()
        if humidity_count < MIN_SAMPLES["humidity"]:
            missing.append(f"humidity ({humidity_count}/{MIN_SAMPLES['humidity']})")
        
        return {"valid": len(missing) == 0, "missing": missing}
    
    def _compute_signal_features(
        self,
        df: pd.DataFrame,
        col: str,
        prefix: str,
        current_name: str,
        ref_time: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute all 21 features for a single signal.
        
        Features:
            - Current value (1)
            - Rolling stats: mean, median, std for 5, 20, 35 min (9)
            - Lagged rolling stats: mean, median, std for 5, 20, 35 min (9)
            - Simple difference (1)
            - 5-minute slope (1)
        """
        features = {}
        
        # Get valid values and timestamps
        mask = df[col].notna()
        values = df.loc[mask, col].values
        timestamps = df.loc[mask, "timestamp"].values
        
        if len(values) < 2:
            return self._empty_signal_features(prefix, current_name)
        
        # ================================================
        # 1. Current value
        # ================================================
        features[current_name] = float(values[-1])
        
        # ================================================
        # 2. Rolling Statistics (5, 20, 35 min windows)
        # ================================================
        for window in self.rolling_windows:
            window_values = self._get_window_values(
                values, timestamps, ref_time, window_minutes=window
            )
            
            if len(window_values) > 0:
                features[f"{prefix}_mean_{window}min"] = float(np.mean(window_values))
                features[f"{prefix}_median_{window}min"] = float(np.median(window_values))
                features[f"{prefix}_std_{window}min"] = float(np.std(window_values)) if len(window_values) > 1 else 0.0
            else:
                features[f"{prefix}_mean_{window}min"] = float(values[-1])
                features[f"{prefix}_median_{window}min"] = float(values[-1])
                features[f"{prefix}_std_{window}min"] = 0.0
        
        # ================================================
        # 3. Lagged Rolling Statistics (10-min resampled)
        # ================================================
        lagged_values, _ = self._resample_to_interval(
            values, timestamps, interval_minutes=LAG_INTERVAL_MINUTES
        )
        
        for window in self.rolling_windows:
            n_samples = window
            
            if len(lagged_values) >= n_samples:
                window_lagged = lagged_values[-n_samples:]
            else:
                window_lagged = lagged_values
            
            if len(window_lagged) > 0:
                features[f"{prefix}_mean_{window}min_lag10m"] = float(np.mean(window_lagged))
                features[f"{prefix}_median_{window}min_lag10m"] = float(np.median(window_lagged))
                features[f"{prefix}_std_{window}min_lag10m"] = float(np.std(window_lagged)) if len(window_lagged) > 1 else 0.0
            else:
                features[f"{prefix}_mean_{window}min_lag10m"] = float(values[-1])
                features[f"{prefix}_median_{window}min_lag10m"] = float(values[-1])
                features[f"{prefix}_std_{window}min_lag10m"] = 0.0
        
        # ================================================
        # 4. Simple Difference (current - previous)
        # ================================================
        features[f"{prefix}_diff_1"] = float(values[-1] - values[-2]) if len(values) >= 2 else 0.0
        
        # ================================================
        # 5. 5-Minute Slope
        # ================================================
        window_5min = self._get_window_values(values, timestamps, ref_time, window_minutes=5)
        
        if len(window_5min) >= 2:
            x = np.arange(len(window_5min))
            slope, _, _, _, _ = stats.linregress(x, window_5min)
            features[f"{prefix}_slope_5m"] = float(slope)
        else:
            features[f"{prefix}_slope_5m"] = 0.0
        
        return features
    
    def _get_window_values(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        ref_time: pd.Timestamp,
        window_minutes: int
    ) -> np.ndarray:
        """Get values within a time window before ref_time."""
        if len(values) == 0:
            return np.array([])
        
        ref_time = pd.Timestamp(ref_time)
        if ref_time.tz is None:
            ref_time = ref_time.tz_localize("UTC")
        
        window_start = ref_time - pd.Timedelta(minutes=window_minutes)
        
        ts = pd.to_datetime(timestamps)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        
        mask = (ts >= window_start) & (ts <= ref_time)
        return values[mask]
    
    def _resample_to_interval(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        interval_minutes: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample values to fixed time intervals (last value per bucket)."""
        if len(values) == 0:
            return np.array([]), np.array([])
        
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps),
            "value": values
        })
        
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        
        df = df.set_index("timestamp")
        resampled = df.resample(f"{interval_minutes}min").last().dropna()
        
        return resampled["value"].values, resampled.index.values
    
    def _empty_signal_features(self, prefix: str, current_name: str) -> Dict[str, float]:
        """Return NaN features for a signal with no data."""
        features = {current_name: np.nan}
        
        for window in self.rolling_windows:
            features[f"{prefix}_mean_{window}min"] = np.nan
            features[f"{prefix}_median_{window}min"] = np.nan
            features[f"{prefix}_std_{window}min"] = np.nan
        
        for window in self.rolling_windows:
            features[f"{prefix}_mean_{window}min_lag10m"] = np.nan
            features[f"{prefix}_median_{window}min_lag10m"] = np.nan
            features[f"{prefix}_std_{window}min_lag10m"] = np.nan
        
        features[f"{prefix}_diff_1"] = np.nan
        features[f"{prefix}_slope_5m"] = np.nan
        
        return features
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to match expected feature order."""
        expected_order = self.get_expected_features()
        ordered_cols = [c for c in expected_order if c in df.columns]
        extra_cols = [c for c in df.columns if c not in expected_order]
        return df[ordered_cols + extra_cols]
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names after transform."""
        return self._feature_names
    
    @staticmethod
    def get_expected_features() -> List[str]:
        """
        Return list of all 86 expected features in order.
        
        84 training features + 2 metadata = 86 total
        """
        features = []
        
        # Skin Temperature (21)
        features.append("temperature")
        for w in ROLLING_WINDOWS:
            features.extend([f"temp_mean_{w}min", f"temp_median_{w}min", f"temp_std_{w}min"])
        for w in ROLLING_WINDOWS:
            features.extend([f"temp_mean_{w}min_lag10m", f"temp_median_{w}min_lag10m", f"temp_std_{w}min_lag10m"])
        features.extend(["temp_diff_1", "temp_slope_5m"])
        
        # Heart Rate (21)
        features.append("bpm")
        for w in ROLLING_WINDOWS:
            features.extend([f"bpm_mean_{w}min", f"bpm_median_{w}min", f"bpm_std_{w}min"])
        for w in ROLLING_WINDOWS:
            features.extend([f"bpm_mean_{w}min_lag10m", f"bpm_median_{w}min_lag10m", f"bpm_std_{w}min_lag10m"])
        features.extend(["bpm_diff_1", "bpm_slope_5m"])
        
        # Ambient Temperature (21)
        features.append("env_Temperature_Celsius")
        for w in ROLLING_WINDOWS:
            features.extend([f"temp_env_mean_{w}min", f"temp_env_median_{w}min", f"temp_env_std_{w}min"])
        for w in ROLLING_WINDOWS:
            features.extend([f"temp_env_mean_{w}min_lag10m", f"temp_env_median_{w}min_lag10m", f"temp_env_std_{w}min_lag10m"])
        features.extend(["temp_env_diff_1", "temp_env_slope_5m"])
        
        # Humidity (21)
        features.append("Relative_Humidity")
        for w in ROLLING_WINDOWS:
            features.extend([f"humidity_env_mean_{w}min", f"humidity_env_median_{w}min", f"humidity_env_std_{w}min"])
        for w in ROLLING_WINDOWS:
            features.extend([f"humidity_env_mean_{w}min_lag10m", f"humidity_env_median_{w}min_lag10m", f"humidity_env_std_{w}min_lag10m"])
        features.extend(["humidity_env_diff_1", "humidity_env_slope_5m"])
        
        # Metadata (2)
        features.extend(["user_id", "timestamp"])
        
        return features
    
    @staticmethod
    def get_training_features() -> List[str]:
        """Return 84 features used for training (excludes metadata)."""
        return [f for f in FeatureTransformer.get_expected_features() if f not in ["user_id", "timestamp"]]


# ============================================
# DATA LOADING UTILITIES
# ============================================

class DataLoader:
    """Load and preprocess raw data files."""
    
    @staticmethod
    def load_fitbit_hr(filepath: str, user_id: str = None) -> pd.DataFrame:
        """Load Fitbit heart rate data."""
        df = DataLoader._load_file(filepath)
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns={"beats per minute": "heart_rate", "bpm": "heart_rate"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
        if user_id:
            df["user_id"] = user_id
        return df[["timestamp", "heart_rate"] + (["user_id"] if user_id else [])]
    
    @staticmethod
    def load_fitbit_skin_temp(filepath: str, user_id: str = None) -> pd.DataFrame:
        """Load Fitbit wrist/skin temperature data."""
        df = DataLoader._load_file(filepath)
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns={"recorded_time": "timestamp", "temperature": "skin_temperature"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["skin_temperature"] = pd.to_numeric(df["skin_temperature"], errors="coerce")
        if user_id:
            df["user_id"] = user_id
        return df[["timestamp", "skin_temperature"] + (["user_id"] if user_id else [])]
    
    @staticmethod
    def load_govee_env(filepath: str, user_id: str = None) -> pd.DataFrame:
        """Load Govee environmental data (drops PM2.5, converts F to C)."""
        df = DataLoader._load_file(filepath)
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns={
            "timestamp for sample frequency every 1 min min": "timestamp",
            "temperature_fahrenheit": "ambient_temp",
            "relative_humidity": "humidity"
        })
        
        # Drop PM2.5
        pm_cols = [c for c in df.columns if "pm2.5" in c.lower()]
        df = df.drop(columns=pm_cols, errors="ignore")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        
        df["ambient_temp"] = pd.to_numeric(df["ambient_temp"], errors="coerce")
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        
        # Convert Fahrenheit to Celsius if needed
        if df["ambient_temp"].median() > 50:
            df["ambient_temp"] = (df["ambient_temp"] - 32) * 5 / 9
        
        if user_id:
            df["user_id"] = user_id
        
        return df[["timestamp", "ambient_temp", "humidity"] + (["user_id"] if user_id else [])]
    
    @staticmethod
    def load_cbt(filepath: str, user_id: str = None) -> pd.DataFrame:
        """Load CBT data (converts Central to UTC, F to C)."""
        df = DataLoader._load_file(filepath)
        df.columns = df.columns.str.lower().str.strip().str.rstrip(":")
        
        # Combine date and time
        if "date" in df.columns and "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), format="mixed")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Central Time to UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("America/Chicago")
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        
        # Find and convert CBT column
        cbt_col = next((c for c in df.columns if any(x in c.lower() for x in ["cbt", "core", "body"])), None)
        if cbt_col is None:
            raise ValueError("CBT column not found")
        
        df["cbt_celsius"] = pd.to_numeric(df[cbt_col], errors="coerce")
        df["cbt_celsius"] = (df["cbt_celsius"] - 32) * 5 / 9
        
        if user_id:
            df["user_id"] = user_id
        
        return df[["timestamp", "cbt_celsius"] + (["user_id"] if user_id else [])]
    
    @staticmethod
    def _load_file(filepath: str) -> pd.DataFrame:
        """Load file with multiple extension support."""
        import os
        ext = os.path.splitext(str(filepath))[1].lower()
        
        if ext in [".csv"]:
            try:
                return pd.read_csv(filepath)
            except:
                return pd.read_csv(filepath, sep="\t")
        elif ext in [".json"]:
            return pd.read_json(filepath)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(filepath)
        return pd.read_csv(filepath)
    
    @staticmethod
    def find_files(directory: str, patterns: List[str] = None) -> List[str]:
        """Find all data files in directory."""
        from pathlib import Path
        patterns = patterns or ["*.csv", "*.CSV", "*.json", "*.JSON", "*.xlsx"]
        files = []
        for pattern in patterns:
            files.extend(Path(directory).glob(pattern))
            files.extend(Path(directory).glob(f"**/{pattern}"))
        return [str(f) for f in files]


# ============================================
# ALIGNMENT UTILITIES
# ============================================

def align_data_to_target(
    fitbit_df: pd.DataFrame,
    env_df: pd.DataFrame,
    target_timestamps: pd.Series,
    lookback_minutes: int = 60
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]]:
    """
    Align input data to target CBT timestamps for training.
    
    Returns list of (fitbit_window, env_window, target_timestamp) tuples.
    Only returns samples where ALL signals meet minimum requirements.
    """
    aligned_data = []
    
    for target_ts in target_timestamps:
        target_ts = pd.Timestamp(target_ts)
        if target_ts.tz is None:
            target_ts = target_ts.tz_localize("UTC")
        
        window_start = target_ts - pd.Timedelta(minutes=lookback_minutes)
        
        # Filter data windows
        fitbit_mask = (fitbit_df["timestamp"] >= window_start) & (fitbit_df["timestamp"] <= target_ts)
        fitbit_window = fitbit_df[fitbit_mask].copy()
        
        env_mask = (env_df["timestamp"] >= window_start) & (env_df["timestamp"] <= target_ts)
        env_window = env_df[env_mask].copy()
        
        # Validate ALL signals have minimum samples
        hr_ok = fitbit_window["heart_rate"].notna().sum() >= MIN_SAMPLES["heart_rate"]
        skin_ok = fitbit_window["skin_temperature"].notna().sum() >= MIN_SAMPLES["skin_temperature"]
        ambient_ok = env_window["ambient_temp"].notna().sum() >= MIN_SAMPLES["ambient_temp"]
        humidity_ok = env_window["humidity"].notna().sum() >= MIN_SAMPLES["humidity"]
        
        if hr_ok and skin_ok and ambient_ok and humidity_ok:
            aligned_data.append((fitbit_window, env_window, target_ts))
    
    return aligned_data