"""End-to-end hub test: start the real HTTP server, POST observations from
two simulated nodes, read the SSE stream, assert the fused output."""

import json
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import numpy as np

# cv2 is only needed by node-side encoding; the hub must not require it.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(__file__).parent / "hub_db"
TMP.mkdir(exist_ok=True)

from hub.protocol import build_observation, encode_vector  # noqa: E402
from hub.server import Handler, Hub, ThreadingHTTPServer  # noqa: E402
from pipeline.stage5_match import ThreatIndex  # noqa: E402
from pipeline.vectors import l2_normalize  # noqa: E402

PORT = 8799
BASE = "http://127.0.0.1:%d" % PORT
D = 512
rng = np.random.default_rng(5)

failures = []


def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + ("   " + detail if detail else ""))
    if not cond:
        failures.append(name)


def post(path, obj):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())


# --- gallery with one enrolled identity ------------------------------------
idx_path, lbl_path = TMP / "h.index", TMP / "h.json"
for p in (idx_path, lbl_path, idx_path.with_suffix(".npy")):
    if p.exists():                 # unlink(missing_ok=) is 3.8+; Nano is 3.6
        p.unlink()

TRUE_A = rng.normal(size=D)
db = ThreatIndex(index_path=idx_path, labels_path=lbl_path, threshold=0.65)
for _ in range(3):
    db.add(l2_normalize(TRUE_A + 0.45 * rng.normal(size=D)), "Subject_A")

hub = Hub()
hub.index = db
hub.fusion.index = db
Handler.hub = hub

server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.4)

# --- SSE subscriber ---------------------------------------------------------
events = []


def listen():
    try:
        resp = urllib.request.urlopen(BASE + "/api/stream", timeout=12)
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    except Exception:
        pass


threading.Thread(target=listen, daemon=True).start()
time.sleep(0.5)

check("health endpoint", get("/api/health")["ok"] is True)
check("dashboard served", b"Cross-view similarity" in
      urllib.request.urlopen(BASE + "/").read())


def observe(node, vector, seq=1):
    payload = build_observation(
        node=node, seq=seq,
        subjects=[{"box": (10, 10, 100, 200), "conf": 0.9,
                   "embedding": l2_normalize(vector), "face_thumb": None}],
        frame_shape=(360, 640, 3), fps=6.0,
        stages_ms={"detect": 20.0, "embed": 8.0},
    )
    return post("/api/observe", payload)


# --- 1. same person on both cameras -> fused match ---------------------------
observe("CAM_A", TRUE_A + 0.5 * rng.normal(size=D))
time.sleep(0.05)
observe("CAM_B", TRUE_A + 0.5 * rng.normal(size=D))
time.sleep(0.4)

state = get("/api/state")
check("two nodes registered", len(state["nodes"]) == 2,
      str(sorted(n["name"] for n in state["nodes"])))
check("similarity reported", state["similarity"] is not None,
      "cross-view %s" % state["similarity"])

mv = [t for t in state["tracks"] if t["multiview"]]
check("fused into one multi-view track", len(mv) == 1,
      "tracks=%d multiview=%d" % (len(state["tracks"]), len(mv)))
if mv:
    t = mv[0]
    check("identity matched", t["is_match"] and t["identity"] == "Subject_A",
          "%s %.3f" % (t["identity"], t["score"]))
    check("fusion gain computed", t["fusion_gain"] is not None,
          "gain %+.3f" % t["fusion_gain"])
    check("both views carry solo scores",
          sum(1 for v in t["views"] if v["solo"]) == 2)
check("alert raised", len(state["alerts"]) >= 1, str(state["alerts"][:1]))
check("in_sync for a sub-150ms gap", state["in_sync"] is True,
      "skew %.1fms vs window %.0fms"
      % (state["skew_ms"], state["settings"]["sync_window_ms"]))

# --- 2. SSE delivered the update --------------------------------------------
check("SSE events received", len(events) >= 2, "%d events" % len(events))
if events:
    check("SSE payload shaped correctly",
          events[-1]["type"] == "update" and "similarity" in events[-1]["data"])

# --- 2b. REGRESSION: a stale partner must not be paired ----------------------
# Wait past max_pair_age, then report from one node only. The hub must treat
# this as single-view rather than fusing against the other node's old frame.
hub.alerts.clear()
hub._last_alert.clear()
time.sleep(1.1)
observe("CAM_A", rng.normal(size=D), seq=99)
time.sleep(0.25)
state = get("/api/state")
check("stale partner rejected", state["similarity"] is None,
      "similarity=%s cameras=%s" % (state["similarity"], state["cameras"]))
check("stale pairing raises no alert", len(state["alerts"]) == 0,
      str(state["alerts"]))

# --- 3. different people -> no fusion ---------------------------------------
hub.alerts.clear()
hub._last_alert.clear()
for n in hub.nodes.values():
    n.buffer.clear()
observe("CAM_A", rng.normal(size=D), seq=2)
time.sleep(0.05)
observe("CAM_B", rng.normal(size=D), seq=2)
time.sleep(0.3)
state = get("/api/state")
mv = [t for t in state["tracks"] if t["multiview"]]
check("strangers are not fused", len(mv) == 0, "%d multiview tracks" % len(mv))
check("similarity still reported for strangers", state["similarity"] is not None,
      "cross-view %s" % state["similarity"])
check("no alert for strangers", len(state["alerts"]) == 0)

# --- 4. settings round-trip --------------------------------------------------
out = post("/api/settings", {"cross_view_threshold": 0.8, "fusion_mode": "max"})
check("settings applied", abs(out["cross_view_threshold"] - 0.8) < 1e-9
      and out["fusion_mode"] == "max")
check("fusion object updated", hub.fusion.cross_view_threshold == 0.8
      and hub.fusion.fusion_mode == "max")
post("/api/settings", {"cross_view_threshold": 0.45, "fusion_mode": "mean"})

# --- 5. a stricter threshold splits a previously fused pair ------------------
post("/api/settings", {"cross_view_threshold": 0.99})
for n in hub.nodes.values():
    n.buffer.clear()
observe("CAM_A", TRUE_A + 0.5 * rng.normal(size=D), seq=3)
time.sleep(0.05)
observe("CAM_B", TRUE_A + 0.5 * rng.normal(size=D), seq=3)
time.sleep(0.3)
state = get("/api/state")
check("threshold 0.99 prevents fusion",
      len([t for t in state["tracks"] if t["multiview"]]) == 0)
post("/api/settings", {"cross_view_threshold": 0.45})

# --- 6. single node only -----------------------------------------------------
hub2 = Hub()
hub2.index = None
hub2.fusion.index = None
obs = build_observation(
    node="SOLO", seq=1,
    subjects=[{"box": (0, 0, 10, 10), "conf": 0.5,
               "embedding": l2_normalize(rng.normal(size=D)), "face_thumb": None}],
    frame_shape=(100, 100, 3))
res = hub2.ingest(obs)
check("single node: no crash, similarity None",
      res["similarity"] is None and len(res["tracks"]) == 1)

# --- 7. clock offset correction ---------------------------------------------
hub3 = Hub()
skewed = build_observation(
    node="LATE", seq=1, subjects=[], frame_shape=(10, 10, 3))
skewed["captured_at"] -= 30.0          # node clock 30s behind the hub
hub3.ingest(skewed)
offset = hub3.nodes["LATE"].clock.offset
check("clock offset estimated", 29.0 < offset < 31.0, "%.2fs" % offset)

# --- 8. malformed input ------------------------------------------------------
try:
    req = urllib.request.Request(BASE + "/api/observe", data=b"{not json",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5)
    check("bad JSON rejected", False, "accepted it")
except urllib.error.HTTPError as e:
    check("bad JSON rejected", e.code == 400, "HTTP %d" % e.code)

hub.ingest({"node": "WEIRD"})           # no subjects, no timestamps
check("minimal payload tolerated", True)

server.shutdown()
print("\n" + ("ALL PASS" if not failures else "%d FAILURE(S): %s" % (len(failures), failures)))
sys.exit(1 if failures else 0)
