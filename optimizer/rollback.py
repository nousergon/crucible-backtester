"""
rollback.py — save and restore previous S3 config files.

Before any optimizer writes new params to S3, it should call save_previous()
to snapshot the current active config. If optimized params cause regression,
rollback() restores the previous version.
"""

import json
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Config type → active S3 key
CONFIG_KEYS = {
    "scoring_weights": "config/scoring_weights.json",
    "executor_params": "config/executor_params.json",
    "predictor_params": "config/predictor_params.json",
    # config#1057 inc 2: MVO optimizer's own params (risk_aversion × tcost_bps),
    # auto-tuned by optimizer/portfolio_optimizer_optimizer.py. Registered here
    # so save_previous() snapshots it before overwrite AND the weekly regression
    # monitor's rollback_all() auto-reverts it if next week's Sortino regresses.
    "portfolio_optimizer": "config/portfolio_optimizer.json",
}


def save_previous(bucket: str, config_type: str) -> bool:
    """
    Copy current active config to {key}_previous.json before overwrite.

    Args:
        bucket: S3 bucket name.
        config_type: one of "scoring_weights", "executor_params", "predictor_params".

    Returns:
        True if previous config was saved, False if no existing config to save.
    """
    active_key = CONFIG_KEYS.get(config_type)
    if not active_key:
        logger.warning("Unknown config type: %s", config_type)
        return False

    previous_key = active_key.replace(".json", "_previous.json")
    s3 = boto3.client("s3")

    try:
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": active_key},
            Key=previous_key,
        )
        logger.info("Saved previous %s to s3://%s/%s", config_type, bucket, previous_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            logger.info("No existing %s in S3 — nothing to save as previous", config_type)
            return False
        raise


def rollback(bucket: str, config_type: str) -> dict:
    """
    Restore previous config by copying _previous.json back to the active key.

    Returns:
        {"rolled_back": True, "config_type": str, "key": str}
        or {"rolled_back": False, "reason": str}
    """
    active_key = CONFIG_KEYS.get(config_type)
    if not active_key:
        return {"rolled_back": False, "reason": f"Unknown config type: {config_type}"}

    previous_key = active_key.replace(".json", "_previous.json")
    s3 = boto3.client("s3")

    try:
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": previous_key},
            Key=active_key,
        )
        logger.info("Rolled back %s from s3://%s/%s", config_type, bucket, previous_key)
        return {"rolled_back": True, "config_type": config_type, "key": active_key}
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return {"rolled_back": False, "reason": f"No previous {config_type} found in S3"}
        return {"rolled_back": False, "reason": str(e)}


def rollback_all(bucket: str) -> list[dict]:
    """Roll back all 3 config types. Returns list of results."""
    results = []
    for config_type in CONFIG_KEYS:
        result = rollback(bucket, config_type)
        results.append(result)
        status = "OK" if result["rolled_back"] else result.get("reason", "failed")
        logger.info("Rollback %s: %s", config_type, status)
    return results
