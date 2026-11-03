import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(samples).encode("utf-8"))
