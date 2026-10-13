"""The live API handler, run offline. Skipped when the model files are not present."""
import base64
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

MODELS = [Path("models/fastseg_large_512x1024.onnx"), Path("models/ufld_culane18_288x800.onnx")]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in MODELS), reason="model files not present")


@pytest.fixture(scope="module")
def handler():
    import handler as h
    return h


def call(h, method="POST", body=None):
    event = {"requestContext": {"http": {"method": method}},
             "body": json.dumps(body) if body is not None else ""}
    r = h.handler(event, None)
    return r["statusCode"], json.loads(r["body"])


def test_browser_visit_shows_the_usage_guide(handler):
    code, body = call(handler, "GET")
    assert code == 200
    assert "OpenCV 5" in body["what_it_does"]


def test_empty_request_is_rejected(handler):
    assert call(handler)[0] == 400


def test_bad_mapillary_id_is_rejected(handler, monkeypatch):
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    assert call(handler, body={"mapillary_id": "abc"})[0] == 400


def test_unusable_photo_is_never_sent_to_bedrock(handler):
    dark = np.full((720, 1280, 3), 15, np.uint8)
    jpg = base64.b64encode(cv.imencode(".jpg", dark)[1].tobytes()).decode()
    code, body = call(handler, body={"image_base64": jpg})
    assert code == 200
    assert body["status"] == "unusable"
    assert body["vlm_answer"].startswith("skipped")
    assert body["opencv_version"].startswith("5")
