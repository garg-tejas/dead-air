"""Minimal HF Spaces health-check app for DispatchR training."""

import http.server
import socketserver
import threading
import time

PORT = 7860

HTML = b"""<!DOCTYPE html>
<html>
<head><title>DispatchR Training</title></head>
<body style="font-family:sans-serif; padding:40px; background:#0f172a; color:white;">
<h1>DispatchR</h1>
<p>Training environment is ready.</p>
<p>Use the <b>Terminal</b> tab to run training commands:</p>
<pre style="background:#1e293b; padding:15px; border-radius:8px;">
python train_unsloth_grpo.py \\
  --model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit \\
  --episodes 200 --batch-size 8 \\
  --push-to-hub --hub-model-id yourname/dispatchr-grpo
</pre>
<p>Or for the manual GRPO fallback:</p>
<pre style="background:#1e293b; padding:15px; border-radius:8px;">
python train_grpo.py \\
  --model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit \\
  --episodes 200 --batch-size 4
</pre>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(HTML)

    def log_message(self, fmt, *args):
        pass  # suppress console spam


def run_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"[DispatchR] Health server running on port {PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    print("[DispatchR] Container ready. Open the Terminal tab to start training.")
    while True:
        time.sleep(3600)
