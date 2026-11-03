"""Simulate a full HTTP POST request to api/predict.py handler."""
import io
import json
import base64
from pathlib import Path
from api.predict import handler

# Load sample image
sample_file = Path("assets/sample_digits/digit_5_sample1_paper.png")
with open(sample_file, "rb") as f:
    b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

payload = json.dumps({"image": b64, "source": "Sample Test"}).encode("utf-8")

# Mock request and socket
class MockSocket:
    def __init__(self, data):
        self.rfile = io.BytesIO(data)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self.rfile
        return self.wfile

request_data = (
    b"POST /api/predict HTTP/1.1\r\n"
    b"Host: localhost\r\n"
    b"Content-Type: application/json\r\n"
    b"Content-Length: " + str(len(payload)).encode("utf-8") + b"\r\n\r\n" + payload
)

class MockServer:
    pass

class MockHandler(handler):
    def log_message(self, format, *args):
        pass

    def log_request(self, code='-', size='-'):
        pass

    def __init__(self, raw_req):
        self.rfile = io.BytesIO(raw_req)
        self.wfile = io.BytesIO()
        self.headers = {
            "Content-Length": str(len(payload)),
            "Content-Type": "application/json"
        }
        self.command = "POST"
        self.path = "/api/predict"
        self.requestline = "POST /api/predict HTTP/1.1"
        self.request_version = "HTTP/1.1"

h = MockHandler(payload)
h.do_POST()

response_output = h.wfile.getvalue().decode("utf-8")
print("Response headers & body:")
print(response_output[:400])

# Parse body
body_start = response_output.find("\r\n\r\n") + 4
body_json = json.loads(response_output[body_start:])
print("\nParsed Result:")
print("Success:", body_json.get("success"))
print("Primary Detected:", body_json.get("primary_detected"))
print("Confidence:", body_json.get("avg_confidence_pct"), "%")
assert body_json.get("success") is True
assert body_json.get("primary_detected") == "5"
print("\n[MOCK HTTP SERVERLESS HANDLER TEST PASSED!]")
