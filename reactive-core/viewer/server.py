"""Read-only HTTP server: GET / (page) and GET /state.json (layer snapshot). Nothing else.
Pure stdlib; runs in a daemon thread inside run.py."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')


def make_handler(snapshot_fn):

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                with open(PAGE, 'rb') as f:
                    self._send(200, f.read(), 'text/html; charset=utf-8')
            elif path == '/state.json':
                body = json.dumps(snapshot_fn(), separators=(',', ':')).encode()
                self._send(200, body, 'application/json')
            else:
                self._send(404, b'not found', 'text/plain')

        def _refuse(self):
            self._send(405, b'read-only viewer', 'text/plain')

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse

        def log_message(self, *a):
            pass

    return H


def serve(snapshot_fn, port, host='0.0.0.0'):
    srv = ThreadingHTTPServer((host, port), make_handler(snapshot_fn))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start_viewer(registry, lock, port):
    def snap():
        with lock:
            return registry.snapshot()
    return serve(snap, port)
