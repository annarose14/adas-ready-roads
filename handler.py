"""AWS Lambda handler - ADAS-Ready Roads analysis endpoint.
POST JSON with ONE of:
  {"image_base64": "<base64 JPEG/PNG>"}     analyse an uploaded road photo
  {"mapillary_id": "1234567890"}            analyse a Mapillary image by id
Returns the frozen OpenCV 5 pipeline verdict as JSON.
CORS headers are added by the Lambda Function URL configuration (not here),
to avoid duplicate Access-Control-Allow-Origin headers that browsers reject."""
import base64
import json
import os
import time

import cv2 as cv
import numpy as np
import requests

from adas_pipeline import PIPELINE_VERSION, RoadAuditor, _result, quality_check, resize, to_record

MODEL_PATH = os.environ.get("MODEL_PATH", "models/fastseg_large_512x1024.onnx")
MAX_BYTES = 6 * 1024 * 1024
AUDITOR = RoadAuditor(MODEL_PATH)          # loaded once per container (engine from DNN_ENGINE, default classic)


def _response(code, payload):
    return {"statusCode": code,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload)}


def _mapillary_image(image_id):
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise ValueError("server has no Mapillary token configured")
    if not str(image_id).isdigit():
        raise ValueError("mapillary_id must be digits only")
    meta = requests.get(f"https://graph.mapillary.com/{image_id}",
                        params={"access_token": token,
                                "fields": "thumb_2048_url,computed_geometry,creator"},
                        timeout=20)
    if meta.status_code != 200:
        raise ValueError(f"Mapillary image {image_id} not found or not accessible")
    info = meta.json()
    img = requests.get(info["thumb_2048_url"], timeout=30)
    img.raise_for_status()
    return img.content, info


def handler(event, context):
    t0 = time.time()
    try:
        body = event.get("body", event) if isinstance(event, dict) else {}
        if isinstance(body, str):
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body).decode()
            body = json.loads(body or "{}")

        source = {}
        if "image_base64" in body:
            raw = base64.b64decode(body["image_base64"])
            source = {"type": "upload"}
        elif "mapillary_id" in body:
            raw, info = _mapillary_image(body["mapillary_id"])
            coords = (info.get("computed_geometry") or {}).get("coordinates", [None, None])
            source = {"type": "mapillary", "id": str(body["mapillary_id"]),
                      "lon": coords[0], "lat": coords[1],
                      "creator": (info.get("creator") or {}).get("username", ""),
                      "licence": "CC BY-SA 4.0"}
        else:
            return _response(400, {"error": "send image_base64 or mapillary_id"})
        if len(raw) > MAX_BYTES:
            return _response(413, {"error": "image too large (max 6 MB)"})

        img = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
        if img is None:
            return _response(400, {"error": "could not decode image"})
        img = resize(img)
        q = quality_check(img)
        res = AUDITOR.analyse(img) if q == "ok" else _result("unusable", reason=q)
        out = to_record(res)
        out.update(quality=q, source=source, pipeline_version=PIPELINE_VERSION,
                   opencv_version=cv.__version__, dnn_engine=AUDITOR.engine,
                   latency_ms=int((time.time() - t0) * 1000))
        print(json.dumps({"event": "analysed", "status": out["status"],
                          "source": source.get("type"), "latency_ms": out["latency_ms"]}))
        return _response(200, out)
    except ValueError as e:
        print(json.dumps({"event": "bad_request", "error": str(e)}))
        return _response(400, {"error": str(e)})
    except Exception as e:
        print(json.dumps({"event": "error", "error": str(e)}))
        return _response(500, {"error": "internal error"})
