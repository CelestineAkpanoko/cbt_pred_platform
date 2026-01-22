"""
Lambda Handler for CBT Prediction API

Production-ready AWS Lambda handler with:
    - POST /predict: Make CBT prediction
    - POST /batch: Batch predictions for multiple users
    - GET /health: Health check endpoint

Deployment:
    - Package with dependencies
    - Deploy via CloudFormation or SAM
    - Model stored in S3

Author: CBT Prediction Platform
Date: January 2026
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global instances (for Lambda warm starts)
_predictor = None
_loader = None


def get_predictor():
    """Get or create predictor instance."""
    global _predictor
    if _predictor is None:
        from .predictor import CBTPredictor
        _predictor = CBTPredictor()
        logger.info("Predictor initialized")
    return _predictor


def get_loader():
    """Get or create Fitbit loader instance."""
    global _loader
    if _loader is None:
        from .fitbit_loader import FitbitLoader
        _loader = FitbitLoader()
        logger.info("FitbitLoader initialized")
    return _loader


def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    Main Lambda handler entry point.
    
    Routes requests to appropriate handlers based on path and method.
    
    Args:
        event: Lambda event (API Gateway format)
        context: Lambda context
        
    Returns:
        API Gateway response dict
    """
    # Handle scheduled events (EventBridge)
    if event.get("source") == "scheduled" or event.get("source") == "aws.events":
        return handle_scheduled_batch()
    
    # Extract HTTP method and path
    # Handle both REST API and HTTP API formats
    request_context = event.get("requestContext", {})
    
    # HTTP API format
    http_info = request_context.get("http", {})
    method = http_info.get("method") or event.get("httpMethod", "GET")
    path = http_info.get("path") or event.get("path", "") or event.get("rawPath", "")
    
    logger.info(f"Request: {method} {path}")
    
    # Route requests
    try:
        if "/health" in path:
            return handle_health()
        
        if "/predict" in path and method == "POST":
            return handle_predict(event)
        
        if "/batch" in path and method == "POST":
            return handle_batch(event)
        
        return response(404, {"error": "Not found", "path": path})
        
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return response(500, {"error": "Internal server error", "message": str(e)})


def handle_health() -> Dict:
    """
    Health check endpoint.
    
    GET /health
    
    Returns model status and system information.
    """
    try:
        predictor = get_predictor()
        health = predictor.health_check()
        
        return response(200, {
            **health,
            "lambda_memory": os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE"),
            "lambda_version": os.environ.get("AWS_LAMBDA_FUNCTION_VERSION")
        })
    except Exception as e:
        return response(503, {
            "status": "unhealthy",
            "error": str(e)
        })


def handle_predict(event: Dict) -> Dict:
    """
    Single prediction endpoint.
    
    POST /predict
    
    Request body options:
    
    Option 1 - Direct data:
    {
        "user_id": "user123",
        "fitbit_data": [
            {"timestamp": "2026-01-13T10:00:00", "heart_rate": 72},
            ...
        ],
        "env_data": [
            {"timestamp": "2026-01-13T10:00:00", "ambient_temp": 22.5, "humidity": 45},
            ...
        ],
        "timestamp": "2026-01-13T10:30:00"
    }
    
    Option 2 - Load from S3:
    {
        "user_id": "user123",
        "date": "2026-01-13"
    }
    
    Response:
    {
        "predicted_cbt": 36.8,
        "confidence": "high",
        "prediction_timestamp": "2026-01-13T10:30:00Z",
        "status": "success"
    }
    """
    # Parse body
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return response(400, {"error": "Invalid JSON in request body"})
    
    user_id = body.get("user_id")
    date = body.get("date")
    timestamp = body.get("timestamp")
    
    # Option 1: Direct data provided
    if "fitbit_data" in body:
        fitbit_data = body["fitbit_data"]
        env_data = body.get("env_data")
        
    # Option 2: Load from S3
    elif user_id:
        loader = get_loader()
        fitbit_data = loader.load_user_data(user_id, date)
        
        if fitbit_data is None:
            return response(404, {
                "error": "No data found",
                "user_id": user_id,
                "date": date
            })
        
        env_data = None  # Environmental data typically included in Fitbit data
        
    else:
        return response(400, {
            "error": "Provide either 'fitbit_data' or 'user_id'"
        })
    
    # Make prediction
    predictor = get_predictor()
    result = predictor.predict(
        fitbit_data=fitbit_data,
        env_data=env_data,
        timestamp=timestamp,
        user_id=user_id
    )
    
    # Save prediction to S3 if successful
    if result.get("status") == "success":
        save_prediction(user_id or "api", result, fitbit_data)
    
    # Return response
    if result.get("status") == "success":
        return response(200, result)
    else:
        return response(400, result)


def handle_batch(event: Dict) -> Dict:
    """
    Batch prediction endpoint.
    
    POST /batch
    
    Request body:
    {
        "user_ids": ["user1", "user2", "user3"],
        "date": "2026-01-13"  // optional
    }
    
    Response:
    {
        "processed": 3,
        "successful": 2,
        "failed": 1,
        "results": [
            {"user_id": "user1", "predicted_cbt": 36.8, "status": "success"},
            {"user_id": "user2", "predicted_cbt": 37.1, "status": "success"},
            {"user_id": "user3", "error": "No data", "status": "error"}
        ]
    }
    """
    # Parse body
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return response(400, {"error": "Invalid JSON in request body"})
    
    user_ids = body.get("user_ids", [])
    date = body.get("date")
    
    if not user_ids:
        return response(400, {"error": "Missing 'user_ids' in request"})
    
    if len(user_ids) > 100:
        return response(400, {"error": "Maximum 100 users per batch"})
    
    # Process users
    results = process_users(user_ids, date)
    
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    return response(200, {
        "processed": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    })


def handle_scheduled_batch() -> Dict:
    """
    Handle scheduled batch processing.
    
    Triggered by EventBridge on schedule.
    Processes all users with available data.
    """
    logger.info("Starting scheduled batch processing")
    
    loader = get_loader()
    user_ids = loader.list_users()
    
    logger.info(f"Found {len(user_ids)} users")
    
    if not user_ids:
        return {"status": "complete", "message": "No users found"}
    
    results = process_users(user_ids)
    
    successful = sum(1 for r in results if r.get("status") == "success")
    
    return {
        "status": "complete",
        "processed": len(results),
        "successful": successful,
        "failed": len(results) - successful
    }


def process_users(user_ids: list, date: Optional[str] = None) -> list:
    """Process predictions for multiple users."""
    loader = get_loader()
    predictor = get_predictor()
    results = []
    
    for user_id in user_ids:
        try:
            # Load data
            fitbit_data = loader.load_user_data(user_id, date)
            
            if fitbit_data is None:
                results.append({
                    "user_id": user_id,
                    "status": "error",
                    "error": "No data available"
                })
                continue
            
            # Predict
            result = predictor.predict(
                fitbit_data=fitbit_data,
                user_id=user_id
            )
            result["user_id"] = user_id
            result["date"] = fitbit_data.get("_meta", {}).get("date")
            
            # Save if successful
            if result.get("status") == "success":
                save_prediction(user_id, result, fitbit_data.get("_meta"))
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}")
            results.append({
                "user_id": user_id,
                "status": "error",
                "error": str(e)
            })
    
    return results


def save_prediction(
    user_id: str,
    result: Dict,
    meta: Optional[Dict] = None
) -> None:
    """Save prediction to S3."""
    bucket = os.environ.get("PREDICTION_BUCKET")
    if not bucket:
        return
    
    prefix = os.environ.get("PREDICTIONS_PREFIX", "predictions/")
    
    ts = datetime.utcnow()
    key = f"{prefix}{user_id}/{ts.strftime('%Y-%m-%d')}/{ts.strftime('%H%M%S')}.json"
    
    record = {
        "timestamp": ts.isoformat() + "Z",
        "user_id": user_id,
        "input_meta": meta,
        **result
    }
    
    try:
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(record, default=str),
            ContentType="application/json"
        )
        logger.info(f"Saved prediction: s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")


def response(status_code: int, body: Dict) -> Dict:
    """Build API Gateway response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps(body, default=str)
    }