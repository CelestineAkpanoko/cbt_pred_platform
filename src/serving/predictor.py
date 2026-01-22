"""
CBT Predictor - Production-Ready Prediction Service

Handles model loading from S3 or local storage and provides
predictions from normalized Fitbit + environmental data.

Features:
    - Model caching for Lambda warm starts
    - S3 and local model loading
    - Input validation
    - Confidence scoring
    - Comprehensive error handling

Author: CBT Prediction Platform
Date: January 2026
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import boto3
import joblib
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global model cache for Lambda warm starts
_model_cache: Dict[str, Any] = {}


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


class InputValidationError(Exception):
    """Raised when input validation fails."""
    pass


class CBTPredictor:
    """
    Production CBT prediction service.
    
    Loads model from S3 or local directory and provides predictions.
    Implements caching for Lambda warm starts.
    
    Usage:
        predictor = CBTPredictor()
        result = predictor.predict(fitbit_data, env_data)
    """
    
    # Required minimum samples for prediction
    MIN_SAMPLES = 10
    
    # Expected features count
    EXPECTED_FEATURES = 84
    
    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        local_dir: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            bucket: S3 bucket name (or PREDICTION_BUCKET env var)
            prefix: S3 prefix for model (or MODEL_PREFIX env var)
            local_dir: Local directory for model (or LOCAL_MODEL_DIR env var)
        """
        self.bucket = bucket or os.environ.get("PREDICTION_BUCKET")
        self.prefix = prefix or os.environ.get("MODEL_PREFIX", "models/")
        self.local_dir = local_dir or os.environ.get("LOCAL_MODEL_DIR")
        
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.fill_values: Dict[str, float] = {}
        
        self._load_model()
        
        logger.info(f"CBTPredictor initialized with {len(self.feature_names)} features")
    
    def _get_cache_key(self) -> str:
        """Generate cache key for model."""
        return f"{self.bucket or 'local'}:{self.prefix or self.local_dir}"
    
    def _load_model(self) -> None:
        """Load model from cache, local, or S3."""
        cache_key = self._get_cache_key()
        
        # Check cache
        if cache_key in _model_cache:
            cached = _model_cache[cache_key]
            self.model = cached["model"]
            self.scaler = cached["scaler"]
            self.feature_names = cached["feature_names"]
            self.metadata = cached["metadata"]
            self.fill_values = cached.get("fill_values", {})
            logger.info("Model loaded from cache")
            return
        
        # Load from source
        if self.local_dir and Path(self.local_dir).exists():
            self._load_from_local()
        elif self.bucket:
            self._load_from_s3()
        else:
            raise ModelLoadError(
                "No model source configured. Set LOCAL_MODEL_DIR or PREDICTION_BUCKET."
            )
        
        # Cache for warm starts
        _model_cache[cache_key] = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "fill_values": self.fill_values
        }
    
    def _load_from_local(self) -> None:
        """Load model from local directory."""
        model_dir = Path(self.local_dir)
        
        # Load model
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        features_path = model_dir / "feature_names.json"
        if features_path.exists():
            with open(features_path) as f:
                data = json.load(f)
                self.feature_names = data.get("feature_names", [])
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                self.fill_values = self.metadata.get("fill_values", {})
        
        logger.info(f"Model loaded from local: {model_dir}")
    
    def _load_from_s3(self) -> None:
        """Load model from S3."""
        s3 = boto3.client("s3")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            try:
                # Download model
                model_key = f"{self.prefix}model.joblib"
                local_model = tmpdir / "model.joblib"
                s3.download_file(self.bucket, model_key, str(local_model))
                self.model = joblib.load(local_model)
                
                # Download scaler
                try:
                    scaler_key = f"{self.prefix}scaler.joblib"
                    local_scaler = tmpdir / "scaler.joblib"
                    s3.download_file(self.bucket, scaler_key, str(local_scaler))
                    self.scaler = joblib.load(local_scaler)
                except Exception:
                    logger.warning("Scaler not found in S3")
                
                # Download feature names
                try:
                    features_key = f"{self.prefix}feature_names.json"
                    local_features = tmpdir / "feature_names.json"
                    s3.download_file(self.bucket, features_key, str(local_features))
                    with open(local_features) as f:
                        self.feature_names = json.load(f).get("feature_names", [])
                except Exception:
                    logger.warning("Feature names not found in S3")
                
                # Download metadata
                try:
                    metadata_key = f"{self.prefix}metadata.json"
                    local_metadata = tmpdir / "metadata.json"
                    s3.download_file(self.bucket, metadata_key, str(local_metadata))
                    with open(local_metadata) as f:
                        self.metadata = json.load(f)
                        self.fill_values = self.metadata.get("fill_values", {})
                except Exception:
                    logger.warning("Metadata not found in S3")
                
            except Exception as e:
                raise ModelLoadError(f"Failed to load model from S3: {e}")
        
        logger.info(f"Model loaded from S3: s3://{self.bucket}/{self.prefix}")
    
    def _validate_input(
        self,
        fitbit_data: Union[List[Dict], pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """Validate and normalize input data."""
        
        # Handle dict with data types (from FitbitLoader)
        if isinstance(fitbit_data, dict):
            if "heart_rate" in fitbit_data:
                # Normalize from loader format
                return self._normalize_loader_data(fitbit_data)
            else:
                fitbit_data = [fitbit_data]
        
        # Convert list to DataFrame
        if isinstance(fitbit_data, list):
            if len(fitbit_data) == 0:
                raise InputValidationError("Empty data provided")
            fitbit_data = pd.DataFrame(fitbit_data)
        
        # Validate DataFrame
        if not isinstance(fitbit_data, pd.DataFrame):
            raise InputValidationError(f"Invalid input type: {type(fitbit_data)}")
        
        if len(fitbit_data) < self.MIN_SAMPLES:
            raise InputValidationError(
                f"Insufficient data: {len(fitbit_data)} samples (minimum: {self.MIN_SAMPLES})"
            )
        
        return fitbit_data
    
    def _normalize_loader_data(self, data: Dict) -> pd.DataFrame:
        """Normalize data from FitbitLoader format."""
        records = []
        
        # Get heart rate data
        hr_data = data.get("heart_rate", [])
        for item in hr_data:
            records.append({
                "timestamp": item.get("timestamp"),
                "heart_rate": item.get("value")
            })
        
        if not records:
            raise InputValidationError("No heart rate data found")
        
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df
    
    def _compute_features(
        self,
        fitbit_df: pd.DataFrame,
        env_df: Optional[pd.DataFrame],
        target_timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """Compute features from input data."""
        # Import transformer
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from features.transformations import FeatureTransformer
        
        transformer = FeatureTransformer()
        
        features = transformer.transform(
            fitbit_df, 
            env_df, 
            target_timestamp
        )
        
        if features is None:
            raise PredictionError("Feature transformation failed - insufficient data")
        
        return features
    
    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input."""
        # Align columns
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = self.fill_values.get(col, 0)
        
        # Select and order columns
        features = features[self.feature_names].copy()
        
        # Fill remaining NaN
        for col in features.columns:
            if features[col].isna().any():
                fill_val = self.fill_values.get(col, 0)
                features[col] = features[col].fillna(fill_val)
        
        # Scale
        if self.scaler is not None:
            return self.scaler.transform(features)
        return features.values
    
    def _compute_confidence(
        self,
        n_samples: int,
        features: pd.DataFrame
    ) -> str:
        """Compute prediction confidence level."""
        # Based on data quantity
        if n_samples >= 60:  # ~1 hour of data
            quantity_score = 1.0
        elif n_samples >= 30:
            quantity_score = 0.7
        elif n_samples >= 15:
            quantity_score = 0.4
        else:
            quantity_score = 0.2
        
        # Based on feature completeness
        non_zero_ratio = (features != 0).sum().sum() / features.size
        completeness_score = min(1.0, non_zero_ratio * 1.5)
        
        # Combined score
        score = 0.6 * quantity_score + 0.4 * completeness_score
        
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def predict(
        self,
        fitbit_data: Union[List[Dict], pd.DataFrame, Dict],
        env_data: Optional[Union[List[Dict], pd.DataFrame]] = None,
        timestamp: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make CBT prediction.
        
        Args:
            fitbit_data: Fitbit sensor data (heart rate, skin temp, etc.)
            env_data: Environmental data (ambient temp, humidity)
            timestamp: Target prediction timestamp (default: latest)
            user_id: User identifier for logging
            
        Returns:
            Prediction result dict with:
                - predicted_cbt: float (Celsius)
                - confidence: str (high/medium/low)
                - prediction_timestamp: str (ISO format)
                - status: str (success/error)
                - error: str (if status=error)
        """
        try:
            # Validate input
            fitbit_df = self._validate_input(fitbit_data)
            
            # Prepare env data
            env_df = None
            if env_data is not None:
                if isinstance(env_data, list):
                    env_df = pd.DataFrame(env_data)
                else:
                    env_df = env_data.copy()
                
                if "timestamp" in env_df.columns:
                    env_df["timestamp"] = pd.to_datetime(env_df["timestamp"])
            
            # Determine target timestamp
            if timestamp:
                target_ts = pd.to_datetime(timestamp)
            elif "timestamp" in fitbit_df.columns:
                target_ts = pd.to_datetime(fitbit_df["timestamp"]).max()
            else:
                target_ts = pd.Timestamp.now(tz="UTC")
            
            # Ensure timestamp is timezone-aware
            if target_ts.tz is None:
                target_ts = target_ts.tz_localize("UTC")
            
            # Compute features
            features = self._compute_features(fitbit_df, env_df, target_ts)
            
            # Prepare for model
            X = self._prepare_features(features)
            
            # Predict
            prediction = self.model.predict(X)[0]
            
            # Compute confidence
            confidence = self._compute_confidence(len(fitbit_df), features)
            
            return {
                "predicted_cbt": round(float(prediction), 2),
                "confidence": confidence,
                "prediction_timestamp": target_ts.isoformat(),
                "user_id": user_id,
                "n_samples": len(fitbit_df),
                "status": "success"
            }
            
        except InputValidationError as e:
            logger.warning(f"Input validation error: {e}")
            return {
                "error": str(e),
                "status": "error",
                "error_type": "validation"
            }
        except PredictionError as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "status": "error",
                "error_type": "prediction"
            }
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return {
                "error": f"Internal error: {str(e)}",
                "status": "error",
                "error_type": "internal"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Return health check information."""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "n_features": len(self.feature_names),
            "scaler_loaded": self.scaler is not None,
            "model_source": "local" if self.local_dir else f"s3://{self.bucket}/{self.prefix}",
            "timestamp": datetime.utcnow().isoformat()
        }