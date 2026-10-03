"""Call the deployed AWS Lambda endpoint with a local photo or a Mapillary image id.
Usage: python invoke_remote.py <function_url> [image_path | mapillary_id]"""
import base64
import json
import sys
import time
from pathlib import Path

import requests

url = sys.argv[1]
arg = sys.argv[2] if len(sys.argv) > 2 else "data/kensington/1043027243798906.jpg"
if arg.isdigit():
    payload = {"mapillary_id": arg}
else:
    payload = {"image_base64": base64.b64encode(Path(arg).read_bytes()).decode()}
t = time.time()
r = requests.post(url, json=payload, timeout=120)
print(f"HTTP {r.status_code} in {int((time.time() - t) * 1000)} ms (round trip from this Mac)")
print(json.dumps(r.json(), indent=2))
