import json
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def log_request(self, code='-', size='-'):
        pass

    def do_POST(self):
        resp = json.dumps({"success": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
