"""Hub server - pairs two capture nodes, fuses them, serves the dashboard.

Run this on the laptop that shows the UI:

    python -m hub.server
    python -m hub.server --port 8765 --open

Then point each capture laptop's node_agent at it. The dashboard is at
http://<this-laptop-ip>:8765/ and updates over Server-Sent Events.

The same dashboard also serves the single-board setup: run
``main_pipeline.py --dashboard 8765`` on the Jetson and it starts this
server in-process and publishes its results to it (Hub.publish), so a
laptop or phone on the same network just opens http://<jetson-ip>:8765/.

Both paths run pipeline/analytics.py, so tracking, visitor
re-identification, journeys, predictions and loitering look identical
either way. Its output rides under the ``scene`` key of every update.

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
from pipeline.analytics import SceneAnalytics, fused_track_payload  # noqa: E402
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


def compose_result(started, cameras, skew_ms, in_sync, similarity, track_payloads,
                   nodes, frames, settings, gallery, stats, alerts, scene):
    """The update the dashboard renders. Shared by Hub._fuse (distributed)
    and main_pipeline.py (single board) so both produce the same shape."""
    return {
        "ts": time.time(),
        "uptime_s": round(time.time() - started, 1),
        "cameras": sorted(cameras),
        "skew_ms": round(skew_ms, 1),
        "in_sync": in_sync,
        "similarity": None if similarity is None else round(similarity, 4),
        "tracks": track_payloads,
        "nodes": nodes,
        "frames": frames,
        "settings": dict(settings),
        "gallery": gallery,
        "stats": stats,
        "alerts": list(alerts)[-12:],
        "scene": scene,
    }


def alert_from_event(ev):
    """Analytics events that warrant an operator alert, in alert-table form."""
    if ev.get("kind") != "loiter":
        return None
    return {"at": ev.get("at") or time.strftime("%H:%M:%S"), "kind": "loiter",
            "identity": ev.get("text", "").split(" has been")[0],
            "camera": ev.get("camera"), "text": ev.get("text"),
            "score": None, "cross_view": None, "cameras": [ev.get("camera")],
            "multiview": False, "agreement": False}


class Hub:
    """Ingests node observations, pairs them, fuses, matches, broadcasts.

    ``fusion`` and ``analytics`` can be passed in so main_pipeline.py can
    share its own objects with the dashboard: then dashboard threshold
    changes act on the pipeline directly.
    """

    def __init__(self, sync_window_ms=None, max_alerts=100, fusion=None,
                 analytics=None, index="load"):
        self.nodes = {}
        self.lock = threading.Lock()
        # Fusion and analytics are stateful and not thread-safe; nodes post
        # concurrently, so each observation is processed under this lock.
        self._process_lock = threading.RLock()
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
            "visitor_threshold": getattr(config, "VISITOR_THRESHOLD", 0.45),
            "loiter_seconds": getattr(config, "LOITER_SECONDS", 30.0),
        }
        self._last_alert = {}

        # The gallery is optional: with no enrolled identities the dashboard
        # still shows live cross-view similarity, which is the headline number.
        self.index = None
        if index != "load":
            self.index = index
        else:
            try:
                from pipeline.stage5_match import get_index
                self.index = get_index()
                print("[hub] gallery: %s" % self.index.stats())
            except Exception as exc:
                print("[hub] gallery unavailable (%s); running in similarity-only mode" % exc)

        if fusion is not None:
            self.fusion = fusion
            self.settings["cross_view_threshold"] = fusion.cross_view_threshold
            self.settings["gallery_threshold"] = fusion.gallery_threshold
            self.settings["fusion_mode"] = fusion.fusion_mode
        else:
            self.fusion = CrossViewFusion(
                index=self.index,
                cross_view_threshold=self.settings["cross_view_threshold"],
                gallery_threshold=self.settings["gallery_threshold"],
                fusion_mode=self.settings["fusion_mode"],
            )
        self.analytics = analytics if analytics is not None else SceneAnalytics()
        if analytics is not None:
            self.settings["visitor_threshold"] = analytics.visitors.threshold
            self.settings["loiter_seconds"] = analytics.loiter_s
        self.analytics.apply_settings(self.settings)
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

        with self._process_lock:
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

            # Track this camera's people and record their faces. The track
            # objects ride along in the payload (never serialised) so that
            # when this observation is used as another node's partner, its
            # people keep the same identities.
            subjects = payload.get("subjects") or []
            tracks = self.analytics.track(
                name, hub_time,
                [tuple(s.get("box") or (0, 0, 0, 0)) for s in subjects],
                [float(s.get("conf") or 0.0) for s in subjects],
                (payload.get("frame_w"), payload.get("frame_h")))
            for s, trk in zip(subjects, tracks):
                vec = decode_vector(s.get("embedding"))
                s["_vec"], s["_trk"] = vec, trk
                if vec is not None:
                    self.analytics.face_result(trk, hub_time, embedding=vec,
                                               thumb=s.get("face_thumb"))

            result = self._fuse(name, hub_time, payload, partner)
            self.last_result = result
        self.broadcast({"type": "update", "data": result})
        return result

    def publish(self, result):
        """Single-board mode: main_pipeline.py did all the work; just serve it."""
        with self._process_lock:
            self.last_result = result
        self.broadcast({"type": "update", "data": result})

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
            for s in pl.get("subjects") or []:
                vec = s["_vec"] if "_vec" in s else decode_vector(s.get("embedding"))
                trk = s.get("_trk")
                cached = False
                if vec is None and trk is not None and trk.embedding is not None:
                    # Face not visible in this frame, but this person's track
                    # has one from earlier: they stay identifiable.
                    vec, cached = trk.embedding, True
                obs = ViewObservation(
                    camera=cam,
                    box=tuple(s.get("box") or (0, 0, 0, 0)),
                    conf=float(s.get("conf") or 0.0),
                    crop=np.zeros((1, 1, 3), np.uint8),   # pixels stay on the node
                    embedding=vec,
                    escalated=bool(s.get("escalated", False)),
                    track=trk,
                    cached=cached,
                )
                obs._thumb = s.get("face_thumb")
                obs_list.append(obs)
            observations[cam] = obs_list

        self._sync_fusion_settings()
        tracks = self.fusion.process(observations) if observations else []
        for ev in self.analytics.resolve(hub_time, tracks):
            alert = alert_from_event(ev)
            if alert is not None:
                self.alerts.append(alert)

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

        track_payloads = [fused_track_payload(t, self.analytics.calibrator) for t in tracks]
        for t, tp in zip(tracks, track_payloads):
            self._maybe_alert(t, tp)

        # Show every camera's latest frame, not just the two being paired.
        frames = {n: st.last_payload.get("frame_thumb")
                  for n, st in self.nodes.items() if st.last_payload and st.online}
        frames.update({cam: pl.get("frame_thumb") for cam, pl in views.items()})

        return compose_result(
            started=self.started,
            cameras=views.keys(),
            skew_ms=skew_ms,
            in_sync=in_sync,
            similarity=best_similarity,
            track_payloads=track_payloads,
            nodes=[n.summary() for n in self.nodes.values()],
            frames=frames,
            settings=self.settings,
            gallery=self.index.stats() if self.index else None,
            stats={"pairs": self.pairs_emitted, "skewed": self.pairs_skewed,
                   "alerts": len(self.alerts)},
            alerts=self.alerts,
            scene=self.analytics.state(hub_time),
        )

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
            "kind": "match",
            "identity": identity,
            "score": payload["score"],
            "confidence": payload.get("confidence"),
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
                   "sync_window_ms", "max_pair_age_ms", "alert_cooldown_s",
                   "visitor_threshold", "loiter_seconds"}
        with self._process_lock:
            for key, value in changes.items():
                if key not in allowed:
                    continue
                self.settings[key] = value if key == "fusion_mode" else float(value)
            self._sync_fusion_settings()
            self.analytics.apply_settings(self.settings)
        self.broadcast({"type": "settings", "data": dict(self.settings)})
        return dict(self.settings)

    def snapshot(self):
        with self._process_lock:
            if self.last_result:
                result = dict(self.last_result)
                if self.nodes:
                    result["nodes"] = [n.summary() for n in self.nodes.values()]
                return result
            return compose_result(
                started=self.started, cameras=[], skew_ms=0.0, in_sync=True,
                similarity=None, track_payloads=[],
                nodes=[n.summary() for n in self.nodes.values()], frames={},
                settings=self.settings,
                gallery=self.index.stats() if self.index else None,
                stats={"pairs": 0, "skewed": 0, "alerts": 0}, alerts=[],
                scene=self.analytics.state(time.time()),
            )


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
