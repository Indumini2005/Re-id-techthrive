"""Hub server - pairs two capture nodes, fuses them, serves the dashboard.

Run this on the laptop that shows the UI:

    python -m hub.server
    python -m hub.server --port 8765 --open

Then point each capture laptop's node_agent at it. The dashboard is at
http://<this-laptop-ip>:8765/ and updates over Server-Sent Events.

Standard library only (plus numpy). No framework, no CDN, no outbound
network access - the page is served entirely from disk, so the whole system
still works on an air-gapped LAN.
"""

import argparse
import json
import queue
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from hub.protocol import (  # noqa: E402
    HUB_DEFAULT_PORT,
    NODE_TIMEOUT_S,
    ClockOffset,
    decode_vector,
    dumps,
    loads,
)
from pipeline.fusion import CrossViewFusion, ViewObservation  # noqa: E402
from pipeline.vectors import cosine  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """http.server.ThreadingHTTPServer, which only exists from Python 3.7.

    Built here from the same two parts so the hub also runs on the Nano's
    Python 3.6. One thread per request, so a held-open SSE stream does not
    block the nodes posting observations.
    """
    daemon_threads = True

# Pairing tolerance for the distributed case. Nodes report a few times per
# second over a LAN, not at camera frame rate, so this is much wider than
# config.SYNC_WINDOW_MS (which governs two cameras wired to one board).
DEFAULT_SYNC_WINDOW_MS = 150.0


class NodeState:
    """Everything the dashboard needs to know about one capture laptop."""

    def __init__(self, name):
        self.name = name
        self.clock = ClockOffset()
        self.buffer = deque(maxlen=30)      # recent (hub_time, payload)
        self.last_seen = 0.0
        self.last_payload = None
        self.frames = 0
        self.fps = 0.0
        self.stages_ms = {}
        self.consumed_seq = -1

    @property
    def online(self):
        return (time.time() - self.last_seen) < NODE_TIMEOUT_S

    def summary(self):
        return {
            "name": self.name,
            "online": self.online,
            "fps": round(self.fps, 1),
            "frames": self.frames,
            "stages_ms": self.stages_ms,
            "clock_offset_ms": round(self.clock.offset * 1000.0, 1),
            "last_seen_ago": round(time.time() - self.last_seen, 1) if self.last_seen else None,
            "subjects": len(self.last_payload.get("subjects", [])) if self.last_payload else 0,
        }


class Hub:
    """Ingests node observations, pairs them, fuses, matches, broadcasts."""

    def __init__(self, sync_window_ms=None, max_alerts=100):
        self.nodes = {}
        self.lock = threading.Lock()
        self.subscribers = []
        self.alerts = deque(maxlen=max_alerts)
        self.started = time.time()
        self.pairs_emitted = 0
        self.pairs_skewed = 0

        self.settings = {
            "cross_view_threshold": config.CROSS_VIEW_THRESHOLD,
            "gallery_threshold": config.MATCH_THRESHOLD,
            "fusion_mode": config.FUSION_MODE,
            # Wider than config.SYNC_WINDOW_MS on purpose. That value assumes
            # two cameras on one machine; here each node sends a few
            # observations per second, so the best achievable pairing is
            # roughly half the node's send interval - 83ms at 6 obs/s. A 50ms
            # window would flag every single pair as skewed.
            "sync_window_ms": sync_window_ms or DEFAULT_SYNC_WINDOW_MS,
            # Hard bound on partner age. Beyond this there is no partner at
            # all, so a node going offline degrades to single-view instead of
            # silently fusing against its last frame forever.
            "max_pair_age_ms": 1000.0,
            "alert_cooldown_s": 10.0,
        }
        self._last_alert = {}

        # The gallery is optional: with no enrolled identities the dashboard
        # still shows live cross-view similarity, which is the headline number.
        self.index = None
        try:
            from pipeline.stage5_match import get_index
            self.index = get_index()
            print("[hub] gallery: %s" % self.index.stats())
        except Exception as exc:
            print("[hub] gallery unavailable (%s); running in similarity-only mode" % exc)

        self.fusion = CrossViewFusion(
            index=self.index,
            cross_view_threshold=self.settings["cross_view_threshold"],
            gallery_threshold=self.settings["gallery_threshold"],
            fusion_mode=self.settings["fusion_mode"],
        )
        self.last_result = None

    # -- subscribers -------------------------------------------------------
    def subscribe(self):
        q = queue.Queue(maxsize=8)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event):
        payload = dumps(event)
        with self.lock:
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # A stalled browser tab must never slow down ingestion.
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass

    # -- ingestion ---------------------------------------------------------
    def ingest(self, payload):
        name = payload.get("node") or "UNKNOWN"
        now = time.time()

        with self.lock:
            node = self.nodes.get(name)
            if node is None:
                node = NodeState(name)
                self.nodes[name] = node
                print("[hub] node joined: %s" % name)

            node.clock.update(payload.get("captured_at", now), now)
            hub_time = node.clock.to_hub_time(payload.get("captured_at", now))

            if node.last_seen:
                dt = now - node.last_seen
                inst = 1.0 / dt if dt > 0 else 0.0
                node.fps = inst if node.fps == 0 else 0.8 * node.fps + 0.2 * inst
            node.last_seen = now
            node.frames += 1
            node.last_payload = payload
            node.stages_ms = payload.get("stages_ms", {})
            node.buffer.append((hub_time, payload))

            partner = self._find_partner(name, hub_time)

        result = self._fuse(name, hub_time, payload, partner)
        self.last_result = result
        self.broadcast({"type": "update", "data": result})
        return result

    def _find_partner(self, name, hub_time):
        """Closest-in-time observation from another node, if one is recent
        enough to describe the same moment."""
        window = self.settings["sync_window_ms"] / 1000.0
        max_age = self.settings["max_pair_age_ms"] / 1000.0
        best = None
        best_dt = None
        for other_name, other in self.nodes.items():
            if other_name == name or not other.buffer:
                continue
            for ts, payload in reversed(other.buffer):
                dt = abs(ts - hub_time)
                if best_dt is None or dt < best_dt:
                    best_dt, best = dt, (other_name, ts, payload)

        # No partner at all beats a stale one: pairing a live frame against a
        # dead node's last observation would report a similarity for two
        # moments minutes apart, and could raise an alert from history.
        if best is None or best_dt > max_age:
            return None
        # Inside max_age but outside the sync window still pairs - the UI can
        # then SAY it is skewed rather than silently showing nothing.
        return best + (best_dt, best_dt <= window)

    def _fuse(self, name, hub_time, payload, partner):
        views = {name: payload}
        skew_ms = 0.0
        in_sync = True
        if partner:
            other_name, _, other_payload, dt, ok = partner
            views[other_name] = other_payload
            skew_ms = dt * 1000.0
            in_sync = ok

        observations = {}
        for cam, pl in views.items():
            obs_list = []
            for s in pl.get("subjects", []):
                vec = decode_vector(s.get("embedding"))
                obs = ViewObservation(
                    camera=cam,
                    box=tuple(s.get("box", (0, 0, 0, 0))),
                    conf=float(s.get("conf", 0.0)),
                    crop=np.zeros((1, 1, 3), np.uint8),   # pixels stay on the node
                    embedding=vec,
                    escalated=bool(s.get("escalated", False)),
                )
                obs._thumb = s.get("face_thumb")
                obs_list.append(obs)
            observations[cam] = obs_list

        self._sync_fusion_settings()
        tracks = self.fusion.process(observations) if observations else []

        # Headline number: best cross-view similarity in this frame pair.
        best_similarity = None
        multiview = [t for t in tracks if t.is_multiview]
        if multiview:
            best_similarity = max(t.cross_view_score for t in multiview)
        elif len(views) > 1:
            best_similarity = self._best_unpaired_similarity(observations)

        self.pairs_emitted += 1
        if not in_sync:
            self.pairs_skewed += 1

        track_payloads = [self._track_payload(t) for t in tracks]
        for t, tp in zip(tracks, track_payloads):
            self._maybe_alert(t, tp)

        return {
            "ts": time.time(),
            "uptime_s": round(time.time() - self.started, 1),
            "cameras": sorted(views.keys()),
            "skew_ms": round(skew_ms, 1),
            "in_sync": in_sync,
            "similarity": None if best_similarity is None else round(best_similarity, 4),
            "tracks": track_payloads,
            "nodes": [n.summary() for n in self.nodes.values()],
            "frames": {cam: pl.get("frame_thumb") for cam, pl in views.items()},
            "settings": dict(self.settings),
            "gallery": self.index.stats() if self.index else None,
            "stats": {
                "pairs": self.pairs_emitted,
                "skewed": self.pairs_skewed,
                "alerts": len(self.alerts),
            },
            "alerts": list(self.alerts)[-12:],
        }

    def _best_unpaired_similarity(self, observations):
        """Both cameras saw somebody, but association rejected the pairing.
        Still report the raw similarity - 'we looked and they differ' is
        information, and a blank readout is not."""
        cams = [c for c, obs in observations.items() if obs]
        if len(cams) < 2:
            return None
        best = None
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                for a in observations[cams[i]]:
                    for b in observations[cams[j]]:
                        if a.embedding is None or b.embedding is None:
                            continue
                        score = cosine(a.embedding, b.embedding)
                        if best is None or score > best:
                            best = score
        return best

    def _track_payload(self, track):
        views = []
        for obs in track.observations:
            views.append({
                "camera": obs.camera,
                "box": list(obs.box),
                "conf": round(obs.conf, 3),
                "escalated": obs.escalated,
                "thumb": getattr(obs, "_thumb", None),
                "solo": {
                    "identity": obs.solo_match.identity,
                    "score": round(obs.solo_match.score, 4),
                    "is_match": obs.solo_match.is_match,
                } if obs.solo_match else None,
            })

        solo_scores = [v["solo"]["score"] for v in views if v["solo"]]
        fused_score = round(track.match.score, 4) if track.match else None
        gain = None
        if fused_score is not None and solo_scores:
            gain = round(fused_score - max(solo_scores), 4)

        return {
            "cameras": track.cameras,
            "n_views": track.n_views,
            "multiview": track.is_multiview,
            "cross_view_score": round(track.cross_view_score, 4),
            "identity": track.match.identity if track.match else None,
            "score": fused_score,
            "is_match": bool(track.match and track.match.is_match),
            "agreement": track.agreement,
            "fusion_gain": gain,
            "notes": list(track.notes),
            "views": views,
        }

    def _maybe_alert(self, track, payload):
        if not payload["is_match"]:
            return
        identity = payload["identity"]
        now = time.time()
        if now - self._last_alert.get(identity, 0) < self.settings["alert_cooldown_s"]:
            return
        self._last_alert[identity] = now
        self.alerts.append({
            "at": time.strftime("%H:%M:%S"),
            "identity": identity,
            "score": payload["score"],
            "cross_view": payload["cross_view_score"],
            "cameras": payload["cameras"],
            "multiview": payload["multiview"],
            "agreement": payload["agreement"],
        })
        print("[hub] MATCH %s  fused %.3f  views %s"
              % (identity, payload["score"], "+".join(payload["cameras"])))

    # -- settings ----------------------------------------------------------
    def _sync_fusion_settings(self):
        self.fusion.cross_view_threshold = self.settings["cross_view_threshold"]
        self.fusion.gallery_threshold = self.settings["gallery_threshold"]
        self.fusion.fusion_mode = self.settings["fusion_mode"]

    def update_settings(self, changes):
        allowed = {"cross_view_threshold", "gallery_threshold", "fusion_mode",
                   "sync_window_ms", "max_pair_age_ms", "alert_cooldown_s"}
        for key, value in changes.items():
            if key not in allowed:
                continue
            self.settings[key] = value if key == "fusion_mode" else float(value)
        self._sync_fusion_settings()
        self.broadcast({"type": "settings", "data": dict(self.settings)})
        return dict(self.settings)

    def snapshot(self):
        if self.last_result:
            result = dict(self.last_result)
            result["nodes"] = [n.summary() for n in self.nodes.values()]
            return result
        return {
            "ts": time.time(),
            "uptime_s": round(time.time() - self.started, 1),
            "cameras": [],
            "similarity": None,
            "tracks": [],
            "nodes": [n.summary() for n in self.nodes.values()],
            "frames": {},
            "settings": dict(self.settings),
            "gallery": self.index.stats() if self.index else None,
            "stats": {"pairs": 0, "skewed": 0, "alerts": 0},
            "alerts": [],
        }


# --- HTTP ------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    hub = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass                                   # the hub prints what matters

    # -- helpers
    def _send(self, code, body=b"", content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, dumps(obj))

    # -- routes
    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            return self._serve_static("dashboard.html", "text/html; charset=utf-8")
        if path == "/api/state":
            return self._json(self.hub.snapshot())
        if path == "/api/stream":
            return self._stream()
        if path == "/api/health":
            return self._json({"ok": True, "uptime_s": round(time.time() - self.hub.started, 1)})
        return self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"

        if path == "/api/observe":
            try:
                payload = loads(raw)
            except Exception as exc:
                return self._json({"error": "bad json: %s" % exc}, 400)
            self.hub.ingest(payload)
            return self._json({"ok": True, "settings": self.hub.settings})

        if path == "/api/settings":
            try:
                changes = loads(raw)
            except Exception as exc:
                return self._json({"error": "bad json: %s" % exc}, 400)
            return self._json(self.hub.update_settings(changes))

        return self._send(404, b'{"error":"not found"}')

    def _serve_static(self, name, content_type):
        path = STATIC_DIR / name
        if not path.exists():
            return self._send(404, b"missing static file")
        return self._send(200, path.read_bytes(), content_type)

    def _stream(self):
        """Server-Sent Events. Chunked encoding is disabled by setting no
        Content-Length and closing on disconnect."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        q = self.hub.subscribe()
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.write(b"data: " + dumps(
                {"type": "update", "data": self.hub.snapshot()}) + b"\n\n")
            self.wfile.flush()
            while True:
                try:
                    payload = q.get(timeout=5.0)
                    self.wfile.write(b"data: " + payload + b"\n\n")
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")   # hold the connection
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.hub.unsubscribe(q)


def local_ip():
    """Best-guess LAN address, for telling the nodes where to connect."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))            # no packets are actually sent
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="Multi-camera hub + dashboard")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=HUB_DEFAULT_PORT)
    ap.add_argument("--sync_window", type=float, default=DEFAULT_SYNC_WINDOW_MS,
                    help="ms between paired observations before flagging skew; "
                         "about half a node's send interval is right")
    ap.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = ap.parse_args()

    Handler.hub = Hub(sync_window_ms=args.sync_window)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    ip = local_ip()
    print("=" * 62)
    print("hub listening on %s:%d" % (args.host, args.port))
    print("dashboard   : http://%s:%d/" % (ip, args.port))
    print("node target : --hub http://%s:%d" % (ip, args.port))
    print("=" * 62)
    print("waiting for capture nodes...\n")

    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open("http://127.0.0.1:%d/" % args.port)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
