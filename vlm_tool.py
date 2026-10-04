"""ADAS-Ready Roads - agent Tool C: vision-language check on Amazon Bedrock.
ask_lines_visible(path, image_id) -> "YES" | "NO" | "UNCLEAR"
Model: Amazon Nova Lite (env VLM_MODEL_ID, default apac.amazon.nova-lite-v1:0), temperature 0.
Only frames that passed the OpenCV 5 gates (quality, road, not own-bonnet) are sent.
Answers are cached in vlm_cache.json (keyed by model + image id) so runs are reproducible and
each photo is paid for once."""
import json
import os
import time
from pathlib import Path

import boto3
import cv2 as cv
from botocore.exceptions import ClientError

MODEL_ID = os.environ.get("VLM_MODEL_ID", "apac.amazon.nova-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
CACHE = Path("vlm_cache.json")
PROMPT = ("You are checking street-level photos for road-marking visibility. "
          "Look only at the road surface. Are painted lane lines (centre line, lane dividers, "
          "or painted edge lines) visible on the road in this photo? Ignore parking-bay lines, "
          "pedestrian crossings and kerbs. Answer with exactly one word: YES, NO, or UNCLEAR "
          "(UNCLEAR if the photo does not show a road from a vehicle's viewpoint).")

_client = None
_cache = None


class VLMBlocked(Exception):
    """AWS refused access (e.g. account verification pending)."""


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def _get_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    return _cache


def _encode(img):
    h, w = img.shape[:2]
    if w > 1024:
        img = cv.resize(img, (1024, int(h * 1024 / w)))
    return cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def ask_image(img):
    """Ask about an in-memory BGR image (no cache). Used by the Lambda endpoint."""
    data = _encode(img)
    for attempt in range(4):
        try:
            resp = _get_client().converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": data}}},
                    {"text": PROMPT}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0})
            text = resp["output"]["message"]["content"][0]["text"].strip().upper()
            return next((w for w in ("YES", "NO", "UNCLEAR") if text.startswith(w)), "UNCLEAR")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException":
                time.sleep(2 * (attempt + 1))
                continue
            if code == "AccessDeniedException":
                raise VLMBlocked(e.response["Error"]["Message"][:160])
            raise
    return "UNCLEAR"


def ask_lines_visible(path, image_id):
    cache = _get_cache()
    key = f"{MODEL_ID}|{image_id}"
    if key in cache:
        return cache[key]
    img = cv.imread(str(path))
    if img is None:
        return "UNCLEAR"
    answer = ask_image(img)
    cache[key] = answer
    CACHE.write_text(json.dumps(cache))
    return answer
