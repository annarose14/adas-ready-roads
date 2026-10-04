"""AWS Lambda handler - ADAS-Ready Roads v5 single-photo endpoint.
GET  (a browser visit)                      -> short usage guide with a link to the dashboard
POST JSON with ONE of:
  {"image_base64": "<base64 JPEG/PNG>"}     analyse an uploaded road photo
  {"mapillary_id": "1234567890"}            analyse a Mapillary image by id
Pipeline: OpenCV 5 quality gate -> segmentation road gate / own-bonnet removal -> lane network +
paint check (pipeline v4) -> ONLY for real road photos: Amazon Bedrock VLM asked
"are painted lane lines visible?" -> lines_visible = yes / no / unknown.
CORS headers are added by the Lambda Function URL configuration (not here)."""
import base64
import json
import os
import time

import cv2 as cv
import numpy as np
import requests

from adas_pipeline import PIPELINE_VERSION, RoadAuditor, _result, quality_check, resize, to_record
from vlm_tool import MODEL_ID as VLM_MODEL, VLMBlocked, ask_image

MAX_BYTES = 6 * 1024 * 1024
ROAD_STATUSES = {"painted", "inferred", "no_lanes"}
DASHBOARD_URL = "https://d1dzzj0gvxxl4o.cloudfront.net"
AUDITOR = RoadAuditor()            # both OpenCV DNN models loaded once per container (classic engine)

USAGE = {
    "service": "ADAS-Ready Roads API (OpenCV AI Competition 2026)",
    "what_it_does": "Checks one road photo with OpenCV 5, then asks a vision language model on "
                    "Amazon Bedrock whether painted lane lines are visible.",
    "try_it_in_a_browser": DASHBOARD_URL + " (use the 'Try it live' box)",
    "how_to_call": "Send a POST request with JSON: {\"mapillary_id\": \"1043027243798906\"} "
                   "or {\"image_base64\": \"<base64 JPEG>\"}",
    "example_curl": "curl -X POST <this URL> -H 'content-type: application/json' "
                    "-d '{\"mapillary_id\": \"1043027243798906\"}'",
    "pipeline_version": PIPELINE_VERSION,
    "agent_version": "v5",
}


def _response(code, payload):
    return {"statusCode": code,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload, indent=2)}


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
        method = ((event.get("requestContext") or {}).get("http") or {}).get("method", "POST") \
            if isinstance(event, dict) else "POST"
        if method == "GET":
            return _response(200, USAGE)

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
            return _response(400, {"error": "send image_base64 or mapillary_id", "help": USAGE})
        if len(raw) > MAX_BYTES:
            return _response(413, {"error": "image too large (max 6 MB)"})

        img = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
        if img is None:
            return _response(400, {"error": "could not decode image"})
        img = resize(img)
        q = quality_check(img)
        res = AUDITOR.analyse(img) if q == "ok" else _result("unusable", reason=q)
        out = to_record(res)
        t_cv = int((time.time() - t0) * 1000)

        vlm = "skipped (not a usable road photo)"
        if res["status"] in ROAD_STATUSES:
            try:
                vlm = ask_image(img).lower()
            except VLMBlocked as e:
                vlm = "unavailable"
                print(json.dumps({"event": "vlm_blocked", "error": str(e)}))
            except Exception as e:  # never fail the whole request because of the VLM
                vlm = "unavailable"
                print(json.dumps({"event": "vlm_error", "error": str(e)[:200]}))
        lines_visible = {"yes": "yes", "no": "no"}.get(vlm, "unknown")

        out.update(quality=q, opencv_status=res["status"], vlm_answer=vlm, lines_visible=lines_visible,
                   source=source, pipeline_version=PIPELINE_VERSION, agent_version="v5",
                   vlm_model=VLM_MODEL, opencv_version=cv.__version__, dnn_engine=AUDITOR.engine,
                   opencv_ms=t_cv, latency_ms=int((time.time() - t0) * 1000))
        print(json.dumps({"event": "analysed", "status": res["status"], "vlm": vlm,
                          "source": source.get("type"), "latency_ms": out["latency_ms"]}))
        return _response(200, out)
    except ValueError as e:
        print(json.dumps({"event": "bad_request", "error": str(e)}))
        return _response(400, {"error": str(e)})
    except Exception as e:
        print(json.dumps({"event": "error", "error": str(e)}))
        return _response(500, {"error": "internal error"})
