"""Send a local road photo to the Lambda container running on this Mac (port 9000)."""
import base64
import json
import sys
from pathlib import Path

import requests

path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(Path("data/parramatta_rd").glob("*.jpg"))
payload = {"image_base64": base64.b64encode(path.read_bytes()).decode()}
r = requests.post("http://localhost:9000/2015-03-31/functions/function/invocations",
                  json=payload, timeout=120)
resp = r.json()
body = json.loads(resp["body"]) if isinstance(resp, dict) and "body" in resp else resp
print(f"Image: {path}")
print(json.dumps(body, indent=2))
