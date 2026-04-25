"""Placeholder app for HF Spaces (training runs via HF Jobs, not Spaces)."""

import http.server
import socketserver

PORT = 7860

HTML = b"""<!DOCTYPE html>
<html>
<head><title>DispatchR</title></head>
<body style="font-family:sans-serif; padding:40px;">
<h1>DispatchR</h1>
<p>This Space is not used for training.</p>
<p>Training runs via <a href="https://huggingface.co/docs/hub/jobs">HF Hub Jobs</a>.</p>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(HTML)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"[DispatchR] Server on port {PORT}")
        httpd.serve_forever()
