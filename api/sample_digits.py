import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def log_request(self, code='-', size='-'):
        pass

    def do_GET(self):
        samples_dir = PROJECT_ROOT / "public" / "assets" / "sample_digits"
        if not samples_dir.exists():
            samples_dir = PROJECT_ROOT / "assets" / "sample_digits"

        samples = []
        if samples_dir.exists():
            for f in sorted(samples_dir.glob("digit_*_paper.png"))[:10]:
                digit_label = f.name.split("_")[1]
                samples.append({
                    "label": digit_label,
                    "filename": f.name,
                    "url": f"/assets/sample_digits/{f.name}"
                })

        resp = json.dumps(samples).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
