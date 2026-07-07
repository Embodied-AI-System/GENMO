"""Browser-based camera preview.

OpenCV reads frames from the camera in a background thread. Each frame is
encoded as JPEG and stored as the latest available image. HTTP clients then
read either one JPEG snapshot or a continuous MJPEG stream made from that
latest-frame cache, so the script does not need a local Qt/OpenCV GUI window.
"""

import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class CameraStream:
    def __init__(
        self,
        camera_id=0,
        width=None,
        height=None,
        fps=30,
        quality=85,
        log_interval=1.0,
    ):
        self.cap = cv2.VideoCapture(camera_id)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        self.quality = int(quality)
        self.period = 1.0 / fps if fps and fps > 0 else 0.0
        self.log_interval = max(float(log_interval), 0.0)

        # A one-frame cache shared by the camera thread and HTTP handler.
        # Keeping only the newest JPEG avoids queue buildup when the browser is
        # slower than the camera.
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.cap.release()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def _capture_loop(self):
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        last_log = time.monotonic()
        frames = 0
        while not self.stop_event.is_set():
            start = time.monotonic()
            ok, frame = self.cap.read()
            if ok:
                # Encode in the capture thread so HTTP requests can send bytes
                # directly without touching OpenCV or the camera device.
                ok, encoded = cv2.imencode(".jpg", frame, params)
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
                    frames += 1

            now = time.monotonic()
            if self.log_interval and now - last_log >= self.log_interval:
                fps = frames / (now - last_log)
                print(f"camera fps: {fps:.1f}", flush=True)
                last_log = now
                frames = 0

            elapsed = time.monotonic() - start
            if self.period > elapsed:
                time.sleep(self.period - elapsed)


class CameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_index()
        elif self.path == "/stream.mjpg":
            self._send_stream()
        elif self.path == "/snapshot.jpg":
            self._send_snapshot()
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        return

    def _send_index(self):
        body = b"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Camera Stream</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    header { padding: 10px 14px; background: #1d1d1d; }
    img { display: block; max-width: 100vw; max-height: calc(100vh - 44px); margin: 0 auto; }
  </style>
</head>
<body>
  <header>Camera Stream</header>
  <img src="/stream.mjpg" alt="camera stream">
</body>
</html>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_snapshot(self):
        jpg = self.server.stream.get_jpeg()
        if jpg is None:
            self.send_error(503, "No camera frame available yet")
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.end_headers()
        self.wfile.write(jpg)

    def _send_stream(self):
        # MJPEG is just a long HTTP response containing repeated JPEG parts.
        # Most browsers can render it directly in an <img> tag.
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            jpg = self.server.stream.get_jpeg()
            if jpg is None:
                time.sleep(0.05)
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

            time.sleep(0.01)


def get_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV camera browser viewer")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    stream = CameraStream(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality,
        log_interval=args.log_interval,
    )
    stream.start()

    server = ThreadingHTTPServer((args.host, args.port), CameraHandler)
    server.stream = stream

    ip = get_lan_ip()
    print(f"Serving camera {args.camera_id}")
    print(f"Local:  http://127.0.0.1:{args.port}")
    print(f"Remote: http://{ip}:{args.port}")
    print("Press Ctrl-C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.shutdown()
        server.server_close()
        stream.stop()


if __name__ == "__main__":
    main()
