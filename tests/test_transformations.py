"""
Tests for Feature Transformations

Tests the FeatureTransformer and DataLoader classes to ensure:
    - Correct feature count (86 total, 84 training)
    - Correct feature names matching specification
    - Proper data validation (all 4 signals required)
    - Backward-looking only (no data leakage)
    - Proper unit conversions (F to C, timezone handling)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.transformations import (
    FeatureTransformer, 
    DataLoader, 
    ROLLING_WINDOWS,
    MIN_SAMPLES
)


def create_test_fitbit_data(
    start_time: datetime = None,
    duration_minutes: int = 60,
    hr_base: float = 70,
    skin_temp_base: float = -2.0
) -> pd.DataFrame:
    """Create test Fitbit data with both heart rate and skin temperature."""
    if start_time is None:
        start_time = datetime(2025, 4, 10, 10, 0, 0)
    
    # Heart rate: ~10 samples per minute
    hr_timestamps = pd.date_range(
        start_time, 
        periods=duration_minutes * 10, 
        freq="6s", 
        tz="UTC"
    )
    hr_values = hr_base + np.random.randn(len(hr_timestamps)) * 5
    
    # Skin temp: 1 sample per minute
    skin_timestamps = pd.date_range(
        start_time, 
        periods=duration_minutes, 
        freq="1min", 
        tz="UTC"
    )
    skin_values = skin_temp_base + np.random.randn(len(skin_timestamps)) * 0.3
    
    # Merge on nearest timestamp
    hr_df = pd.DataFrame({
        "timestamp": hr_timestamps,
        "heart_rate": hr_values
    })
    
    skin_df = pd.DataFrame({
        "timestamp": skin_timestamps,
        "skin_temperature": skin_values
    })
    
    # Merge
    combined = pd.merge_asof(
        hr_df.sort_values("timestamp"),
        skin_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=2)
    )
    
    return combined


def create_test_env_data(
    start_time: datetime = None,
    duration_minutes: int = 60,
    temp_base: float = 21.0,
    humidity_base: float = 50.0
) -> pd.DataFrame:
    """Create test environmental data."""
    if start_time is None:
        start_time = datetime(2025, 4, 10, 10, 0, 0)
    
    timestamps = pd.date_range(
        start_time, 
        periods=duration_minutes, 
        freq="1min", 
        tz="UTC"
    )
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "ambient_temp": temp_base + np.random.randn(len(timestamps)) * 0.5,
        "humidity": humidity_base + np.random.randn(len(timestamps)) * 5
    })


def test_feature_count():
    """Test that transformer produces exactly 86 features."""
    print("Testing feature count...")
    
    fitbit_df = create_test_fitbit_data(duration_minutes=60)
    env_df = create_test_env_data(duration_minutes=60)
    
    transformer = FeatureTransformer()
    features = transformer.transform(fitbit_df, env_df, user_id="test_user")
    
    assert features is not None, "Transform returned None"
    assert len(features.columns) == 86, f"Expected 86 features, got {len(features.columns)}"
    
    # Check training features
    training_features = FeatureTransformer.get_training_features()
    assert len(training_features) == 84, f"Expected 84 training features, got {len(training_features)}"
    
    print(f"✓ Feature count correct: {len(features.columns)} total, {len(training_features)} training")


def test_feature_names():
    """Test that all expected feature names are present."""
    print("Testing feature names...")
    
    fitbit_df = create_test_fitbit_data(duration_minutes=60)
    env_df = create_test_env_data(duration_minutes=60)
    
    transformer = FeatureTransformer()
    features = transformer.transform(fitbit_df, env_df, user_id="test_user")
    
    expected_features = FeatureTransformer.get_expected_features()
    
    # Check all expected features exist
    missing = [f for f in expected_features if f not in features.columns]
    extra = [f for f in features.columns if f not in expected_features]
    
    assert len(missing) == 0, f"Missing features: {missing}"
    assert len(extra) == 0, f"Extra features: {extra}"
    
    # Verify specific feature names
    assert "temperature" in features.columns, "Missing 'temperature' (skin temp current)"
    assert "bpm" in features.columns, "Missing 'bpm' (heart rate current)"
    assert "env_Temperature_Celsius" in features.columns, "Missing 'env_Temperature_Celsius'"
    assert "Relative_Humidity" in features.columns, "Missing 'Relative_Humidity'"
    assert "user_id" in features.columns, "Missing 'user_id'"
    assert "timestamp" in features.columns, "Missing 'timestamp'"
    
    # Check rolling feature names
    for window in ROLLING_WINDOWS:
        assert f"temp_mean_{window}min" in features.columns
        assert f"bpm_mean_{window}min" in features.columns
        assert f"temp_env_mean_{window}min" in features.columns
        assert f"humidity_env_mean_{window}min" in features.columns
    
    # Check lagged feature names
    for window in ROLLING_WINDOWS:
        assert f"temp_mean_{window}min_lag10m" in features.columns
        assert f"bpm_mean_{window}min_lag10m" in features.columns
    
    # Check diff and slope
    assert "temp_diff_1" in features.columns
    assert "temp_slope_5m" in features.columns
    assert "bpm_diff_1" in features.columns
    assert "bpm_slope_5m" in features.columns
    
    print(f"✓ All {len(expected_features)} feature names correct")


def test_all_signals_required():
    """Test that transformer requires all 4 signals."""
    print("Testing all signals required...")
    
    transformer = FeatureTransformer()
    start_time = datetime(2025, 4, 10, 10, 0, 0)
    
    # Test with missing skin temperature
    fitbit_hr_only = pd.DataFrame({
        "timestamp": pd.date_range(start_time, periods=600, freq="6s", tz="UTC"),
        "heart_rate": 70 + np.random.randn(600) * 5
    })
    
    env_df = create_test_env_data(duration_minutes=60, start_time=start_time)
    
    try:
        result = transformer.transform(fitbit_hr_only, env_df)
        # Should fail due to missing skin_temperature
        assert False, "Should have raised ValueError for missing skin_temperature"
    except ValueError as e:
        assert "skin_temperature" in str(e).lower()
        print("  ✓ Correctly rejected missing skin_temperature")
    
    # Test with missing heart rate
    fitbit_skin_only = pd.DataFrame({
        "timestamp": pd.date_range(start_time, periods=60, freq="1min", tz="UTC"),
        "skin_temperature": -2 + np.random.randn(60) * 0.3
    })
    
    try:
        result = transformer.transform(fitbit_skin_only, env_df)
        assert False, "Should have raised ValueError for missing heart_rate"
    except ValueError as e:
        assert "heart_rate" in str(e).lower()
        print("  ✓ Correctly rejected missing heart_rate")
    
    # Test with missing environmental data
    fitbit_df = create_test_fitbit_data(duration_minutes=60, start_time=start_time)
    
    result = transformer.transform(fitbit_df, None)
    assert result is None, "Should return None for missing environmental data"
    print("  ✓ Correctly rejected missing environmental data")
    
    print("✓ All signal validation working correctly")


def test_minimum_samples():
    """Test that transformer enforces minimum sample counts."""
    print("Testing minimum sample requirements...")
    
    transformer = FeatureTransformer()
    start_time = datetime(2025, 4, 10, 10, 0, 0)
    
    # Create data with too few heart rate samples
    fitbit_insufficient = pd.DataFrame({
        "timestamp": pd.date_range(start_time, periods=10, freq="6s", tz="UTC"),
        "heart_rate": 70 + np.random.randn(10) * 5,
        "skin_temperature": [-2.0] * 10
    })
    
    env_df = create_test_env_data(duration_minutes=60, start_time=start_time)
    
    result = transformer.transform(fitbit_insufficient, env_df)
    assert result is None, f"Should return None for insufficient HR samples (need {MIN_SAMPLES['heart_rate']})"
    
    print(f"✓ Minimum sample validation working (HR needs {MIN_SAMPLES['heart_rate']} samples)")


def test_backward_looking():
    """Test that features only use data before target timestamp."""
    print("Testing backward-looking constraint...")
    
    start_time = datetime(2025, 4, 10, 10, 0, 0)
    
    # Create data with increasing values
    timestamps = pd.date_range(start_time, periods=100, freq="1min", tz="UTC")
    hr_values = list(range(60, 160))  # 60, 61, 62, ..., 159
    
    fitbit_df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": hr_values,
        "skin_temperature": [-2.0] * 100
    })
    
    env_df = pd.DataFrame({
        "timestamp": timestamps,
        "ambient_temp": [21.0] * 100,
        "humidity": [50.0] * 100
    })
    
    transformer = FeatureTransformer()
    
    # Get features at midpoint (index 50, timestamp 10:50:00)
    target_ts = timestamps[50]
    features = transformer.transform(fitbit_df, env_df, target_timestamp=target_ts)
    
    assert features is not None, "Transform returned None"
    
    # Current HR should be value at index 50 = 110
    assert features["bpm"].iloc[0] == 110, f"Expected bpm=110, got {features['bpm'].iloc[0]}"
    
    # 5-min mean should be based on values 105-110 (indices 45-50)
    # Mean of [105, 106, 107, 108, 109, 110] = 107.5
    expected_mean = np.mean([105, 106, 107, 108, 109, 110])
    actual_mean = features["bpm_mean_5min"].iloc[0]
    
    # Should NOT include future values (111-159)
    assert actual_mean < 115, f"bpm_mean_5min too high ({actual_mean}), likely using future data"
    
    print(f"✓ Backward-looking verified: bpm={features['bpm'].iloc[0]}, mean_5min={actual_mean:.1f}")


def test_data_loader_govee():
    """Test Govee data loading with unit conversion."""
    print("Testing Govee data loader...")
    
    # Create sample Govee CSV
    govee_data = """Timestamp for sample frequency every 1 min min,PM2.5(µg/m³),Temperature_Fahrenheit,Relative_Humidity
2025-04-07 16:51:00,0,69.98,50.6
2025-04-07 16:52:00,2,70.34,48.2
2025-04-07 16:53:00,3,70.16,48.2
2025-04-07 16:54:00,3,70.16,48.0
2025-04-07 16:55:00,4,70.34,48.1
"""
    
    temp_file = Path("temp_govee_test.csv")
    temp_file.write_text(govee_data)
    
    try:
        df = DataLoader.load_govee_env(str(temp_file), user_id="test_user")
        
        # Check PM2.5 is dropped
        assert "pm2.5" not in "".join(df.columns).lower(), "PM2.5 column should be dropped"
        
        # Check temperature converted to Celsius
        # 69.98°F = (69.98 - 32) * 5/9 = 21.1°C
        expected_celsius = (69.98 - 32) * 5 / 9
        assert abs(df["ambient_temp"].iloc[0] - expected_celsius) < 0.1, \
            f"Expected {expected_celsius:.2f}°C, got {df['ambient_temp'].iloc[0]:.2f}°C"
        
        # Check columns
        assert "timestamp" in df.columns
        assert "ambient_temp" in df.columns
        assert "humidity" in df.columns
        assert "user_id" in df.columns
        
        print(f"  ✓ Temperature converted: 69.98°F → {df['ambient_temp'].iloc[0]:.2f}°C")
        print(f"  ✓ PM2.5 dropped, columns: {list(df.columns)}")
        
    finally:
        temp_file.unlink()
    
    print("✓ Govee loader working correctly")


def test_data_loader_cbt():
    """Test CBT data loading with timezone and unit conversion."""
    print("Testing CBT data loader...")
    
    # Create sample CBT CSV
    cbt_data = """Date:,Time:,CBT (Deg F):
4/5/2025,8:59 PM,98.5
4/5/2025,9:09 PM,98.7
4/5/2025,9:19 PM,98.6
"""
    
    temp_file = Path("temp_cbt_test.csv")
    temp_file.write_text(cbt_data)
    
    try:
        df = DataLoader.load_cbt(str(temp_file), user_id="test_user")
        
        # Check timestamp is UTC
        assert df["timestamp"].dt.tz is not None, "Timestamp should have timezone"
        assert str(df["timestamp"].dt.tz) == "UTC", "Timestamp should be UTC"
        
        # Check temperature converted to Celsius
        # 98.5°F = (98.5 - 32) * 5/9 = 36.944°C
        expected_celsius = (98.5 - 32) * 5 / 9
        assert abs(df["cbt_celsius"].iloc[0] - expected_celsius) < 0.01, \
            f"Expected {expected_celsius:.2f}°C, got {df['cbt_celsius'].iloc[0]:.2f}°C"
        
        # Check columns
        assert "timestamp" in df.columns
        assert "cbt_celsius" in df.columns
        assert "user_id" in df.columns
        
        print(f"  ✓ Temperature converted: 98.5°F → {df['cbt_celsius'].iloc[0]:.2f}°C")
        print(f"  ✓ Timezone converted: Central → UTC")
        
    finally:
        temp_file.unlink()
    
    print("✓ CBT loader working correctly")


def test_data_loader_fitbit_hr():
    """Test Fitbit heart rate data loading."""
    print("Testing Fitbit heart rate loader...")
    
    hr_data = """timestamp,beats per minute
2025-04-10T00:00:01Z,67
2025-04-10T00:00:04Z,67
2025-04-10T00:00:05Z,68
2025-04-10T00:00:08Z,68
2025-04-10T00:00:09Z,69
"""
    
    temp_file = Path("temp_hr_test.csv")
    temp_file.write_text(hr_data)
    
    try:
        df = DataLoader.load_fitbit_hr(str(temp_file), user_id="test_user")
        
        assert "timestamp" in df.columns
        assert "heart_rate" in df.columns
        assert df["timestamp"].dt.tz is not None
        assert df["heart_rate"].iloc[0] == 67
        
        print(f"  ✓ Loaded {len(df)} HR samples")
        print(f"  ✓ Column 'beats per minute' → 'heart_rate'")
        
    finally:
        temp_file.unlink()
    
    print("✓ Fitbit HR loader working correctly")


def test_data_loader_fitbit_skin_temp():
    """Test Fitbit skin temperature data loading."""
    print("Testing Fitbit skin temperature loader...")
    
    skin_data = """recorded_time,temperature
2025-05-30T00:00,-2.095291443
2025-05-30T00:01,-2.135291443
2025-05-30T00:02,-2.225291443
"""
    
    temp_file = Path("temp_skin_test.csv")
    temp_file.write_text(skin_data)
    
    try:
        df = DataLoader.load_fitbit_skin_temp(str(temp_file), user_id="test_user")
        
        assert "timestamp" in df.columns
        assert "skin_temperature" in df.columns
        assert df["timestamp"].dt.tz is not None
        assert abs(df["skin_temperature"].iloc[0] - (-2.095291443)) < 0.0001
        
        print(f"  ✓ Loaded {len(df)} skin temp samples")
        print(f"  ✓ Column 'recorded_time' → 'timestamp', 'temperature' → 'skin_temperature'")
        
    finally:
        temp_file.unlink()
    
    print("✓ Fitbit skin temp loader working correctly")


def test_rolling_window_calculation():
    """Test that rolling window calculations are correct."""
    print("Testing rolling window calculations...")
    
    start_time = datetime(2025, 4, 10, 10, 0, 0)
    
    # Create constant data for easy verification
    timestamps = pd.date_range(start_time, periods=60, freq="1min", tz="UTC")
    
    fitbit_df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": [100] * 60,  # Constant 100 bpm
        "skin_temperature": [-2.0] * 60  # Constant -2.0
    })
    
    env_df = pd.DataFrame({
        "timestamp": timestamps,
        "ambient_temp": [21.0] * 60,  # Constant 21°C
        "humidity": [50.0] * 60  # Constant 50%
    })
    
    transformer = FeatureTransformer()
    features = transformer.transform(fitbit_df, env_df)
    
    # With constant values, mean/median should equal the constant
    assert features["bpm"].iloc[0] == 100
    assert features["bpm_mean_5min"].iloc[0] == 100
    assert features["bpm_median_5min"].iloc[0] == 100
    assert features["bpm_std_5min"].iloc[0] == 0  # No variation
    
    assert features["temperature"].iloc[0] == -2.0
    assert features["temp_mean_5min"].iloc[0] == -2.0
    
    # Diff should be 0 for constant values
    assert features["bpm_diff_1"].iloc[0] == 0
    assert features["temp_diff_1"].iloc[0] == 0
    
    # Slope should be 0 for constant values
    assert features["bpm_slope_5m"].iloc[0] == 0
    assert features["temp_slope_5m"].iloc[0] == 0
    
    print("✓ Rolling window calculations correct")


def test_metadata_features():
    """Test that metadata features are correctly populated."""
    print("Testing metadata features...")
    
    fitbit_df = create_test_fitbit_data(duration_minutes=60)
    env_df = create_test_env_data(duration_minutes=60)
    
    transformer = FeatureTransformer()
    
    # Test with user_id
    features = transformer.transform(fitbit_df, env_df, user_id="test_user_123")
    
    assert features["user_id"].iloc[0] == "test_user_123"
    assert features["timestamp"].iloc[0] is not None
    
    # Test without user_id
    features_no_user = transformer.transform(fitbit_df, env_df)
    assert features_no_user["user_id"].iloc[0] == "unknown"
    
    print("✓ Metadata features correct")


def test_get_training_features():
    """Test that get_training_features excludes metadata."""
    print("Testing get_training_features()...")
    
    training_features = FeatureTransformer.get_training_features()
    
    assert "user_id" not in training_features, "user_id should not be in training features"
    assert "timestamp" not in training_features, "timestamp should not be in training features"
    assert len(training_features) == 84, f"Expected 84 training features, got {len(training_features)}"
    
    print(f"✓ Training features: {len(training_features)} (excludes user_id, timestamp)")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FEATURE TRANSFORMATIONS")
    print("=" * 60)
    print()
    
    test_feature_count()
    print()
    
    test_feature_names()
    print()
    
    test_all_signals_required()
    print()
    
    test_minimum_samples()
    print()
    
    test_backward_looking()
    print()
    
    test_data_loader_govee()
    print()
    
    test_data_loader_cbt()
    print()
    
    test_data_loader_fitbit_hr()
    print()
    
    test_data_loader_fitbit_skin_temp()
    print()
    
    test_rolling_window_calculation()
    print()
    
    test_metadata_features()
    print()
    
    test_get_training_features()
    print()
    
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)