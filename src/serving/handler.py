"""
Lambda Handler for CBT Prediction API

Production-ready AWS Lambda handler with:
    - POST /predict: Make CBT prediction from 84 features
    - GET /health: Health check endpoint
    - Direct model loading (no predictor class)

Author: CBT Prediction Platform
Date: January 2026
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global model cache for Lambda warm starts
_model_cache: Dict[str, Any] = {
    "model": None,
    "scaler": None,
    "feature_names": None,
    "loaded": False
}


def load_model() -> Dict[str, Any]:
    """Load model, scaler, and feature names from local directory or S3."""
    try:
        import joblib
        from pathlib import Path
        
        # Check for local model first (for Docker/local testing)
        local_dir = os.environ.get("LOCAL_MODEL_DIR")
        logger.info(f"LOCAL_MODEL_DIR env var: {local_dir}")
        
        if local_dir:
            model_path = Path(local_dir)
            logger.info(f"Model path: {model_path}")
            logger.info(f"Model path exists: {model_path.exists()}")
            
            if model_path.exists():
                # List all files for debugging
                try:
                    files = list(model_path.iterdir())
                    logger.info(f"Files in model dir: {[f.name for f in files]}")
                except Exception as e:
                    logger.error(f"Error listing directory: {e}")
            
            model_file = model_path / "model.joblib"
            scaler_file = model_path / "scaler.joblib"
            features_file = model_path / "feature_names.json"
            
            logger.info(f"model.joblib exists: {model_file.exists()}")
            logger.info(f"scaler.joblib exists: {scaler_file.exists()}")
            logger.info(f"feature_names.json exists: {features_file.exists()}")
            
            if model_file.exists():
                logger.info(f"Loading model from local: {model_path}")
                _model_cache["model"] = joblib.load(model_file)
                logger.info("Model loaded successfully")
                
                if scaler_file.exists():
                    _model_cache["scaler"] = joblib.load(scaler_file)
                    logger.info("Scaler loaded successfully")
                else:
                    logger.warning("scaler.joblib not found - using None")
                    _model_cache["scaler"] = None
                
                if features_file.exists():
                    with open(features_file) as f:
                        data = json.load(f)
                        # Extract just the list of feature names
                        if isinstance(data, dict):
                            _model_cache["feature_names"] = data.get("feature_names", [])
                        elif isinstance(data, list):
                            _model_cache["feature_names"] = data
                        else:
                            _model_cache["feature_names"] = []
                    logger.info(f"Feature names loaded: {len(_model_cache['feature_names'])} features")
                    logger.info(f"First 3 features: {_model_cache['feature_names'][:3]}")
                else:
                    logger.warning("feature_names.json not found")
                    _model_cache["feature_names"] = []
                
                _model_cache["loaded"] = True
                return {"success": True}
            else:
                logger.error(f"model.joblib not found at {model_file}")
                return {"success": False, "error": f"model.joblib not found at {model_file}"}
        
        # Fall back to S3 (for production)
        import boto3
        
        bucket = os.environ.get("MODEL_BUCKET")
        prefix = os.environ.get("MODEL_PREFIX", "models/")
        
        if not bucket:
            return {"success": False, "error": "No local model found and MODEL_BUCKET not configured"}
        
        s3 = boto3.client("s3")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download model files
            files_to_download = {
                "model": f"{prefix}model.joblib",
                "scaler": f"{prefix}scaler.joblib",
                "features": f"{prefix}feature_names.json"
            }
            
            for name, key in files_to_download.items():
                local_path = f"{tmpdir}/{name}"
                try:
                    s3.download_file(bucket, key, local_path)
                except Exception as e:
                    return {"success": False, "error": f"Failed to download {key}: {e}"}
            
            # Load into memory
            _model_cache["model"] = joblib.load(f"{tmpdir}/model")
            _model_cache["scaler"] = joblib.load(f"{tmpdir}/scaler")
            
            with open(f"{tmpdir}/features", "r") as f:
                data = json.load(f)
                # Extract just the list of feature names
                if isinstance(data, dict):
                    _model_cache["feature_names"] = data.get("feature_names", [])
                elif isinstance(data, list):
                    _model_cache["feature_names"] = data
                else:
                    _model_cache["feature_names"] = []
            
            _model_cache["loaded"] = True
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def predict_from_features(features: Dict[str, float]) -> Dict[str, Any]:
    """Make prediction from pre-computed 84 features."""
    try:
        import numpy as np
        
        # Get expected feature names
        expected_features = _model_cache.get("feature_names")
        
        if expected_features is None or len(expected_features) == 0:
            logger.error("Feature names not loaded in cache")
            return {"error": "Model feature names not available"}
        
        logger.info(f"Expected features: {len(expected_features)}")
        logger.info(f"Received features: {len(features)}")
        
        # Check for missing features
        missing = [f for f in expected_features if f not in features]
        
        if missing:
            logger.error(f"Missing {len(missing)} features: {missing[:5]}")
            return {
                "error": "Missing features",
                "missing_count": len(missing),
                "missing": missing[:10],
                "expected_count": len(expected_features),
                "received_count": len(features)
            }
        
        # Build feature array in correct order
        feature_array = np.array([features[f] for f in expected_features]).reshape(1, -1)
        logger.info(f"Feature array shape: {feature_array.shape}")
        
        # Scale features if scaler exists
        if _model_cache.get("scaler") is not None:
            feature_array = _model_cache["scaler"].transform(feature_array)
            logger.info("Features scaled")
        
        # Make prediction
        model = _model_cache.get("model")
        if model is None:
            return {"error": "Model not loaded"}
        
        prediction = model.predict(feature_array)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Convert to Fahrenheit
        prediction_f = (prediction * 9/5) + 32
        
        return {
            "prediction": {
                "cbt_celsius": round(float(prediction), 2),
                "cbt_fahrenheit": round(float(prediction_f), 2)
            },
            "features_used": len(expected_features),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return {"error": f"Prediction failed: {str(e)}"}


def handle_health() -> Dict[str, Any]:
    """Health check endpoint."""
    # Try to load model if not loaded
    if not _model_cache["loaded"]:
        load_result = load_model()
        if not load_result["success"]:
            logger.error(f"Health check: model load failed - {load_result.get('error')}")
    
    return response(200, {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": _model_cache["loaded"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


def handle_predict(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle prediction request."""
    # Ensure model is loaded
    if not _model_cache["loaded"]:
        load_result = load_model()
        if not load_result["success"]:
            return response(503, {"error": "Model not loaded", "details": load_result.get("error")})
    
    # Parse request body
    body = parse_body(event)
    if isinstance(body, dict) and "error" in body:
        return response(400, body)
    
    # Get features from request
    features = body.get("features")
    if not features:
        return response(400, {"error": "Missing 'features' in request body"})
    
    # Make prediction
    result = predict_from_features(features)
    
    if "error" in result:
        return response(400, result)
    
    return response(200, result)


def parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Parse request body from event."""
    body = event.get("body", "{}")
    
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in request body"}
    
    return body


def response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Build API Gateway response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body, default=str)
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda entry point."""
    try:
        path = event.get("rawPath", "/")
        method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
        
        logger.info(f"{method} {path}")
        
        # Route requests
        if path == "/health":
            return handle_health()
        elif path == "/predict":
            return handle_predict(event)
        else:
            return response(404, {"error": "Not found"})
            
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return response(500, {"error": str(e)})