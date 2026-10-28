"""Scene analytics end to end, on synthetic people.

Each simulated person has a true face direction; every sighting is that
direction plus noise, tuned so two live looks at the same person score
about 0.6 and two different people about 0. The harness runs exactly the
per-cycle sequence main_pipeline.py runs: track -> face check -> fusion ->
resolve.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.analytics import SceneAnalytics, fused_track_payload  # noqa: E402
from pipeline.calibration import ScoreCalibrator, build_calibration, fit_logistic  # noqa: E402
from pipeline.fusion import CrossViewFusion, ViewObservation  # noqa: E402
from pipeline.stage5_match import ThreatIndex  # noqa: E402
from pipeline.vectors import l2_normalize  # noqa: E402

TMP = Path(__file__).parent / "analytics_tmp"
TMP.mkdir(exist_ok=True)
D = 512
DT = 1.0 / 3.0
rng = np.random.default_rng(21)
failures = []


def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + ("   " + detail if detail else ""))
    if not cond:
        failures.append(name)


def new_person():
    return l2_normalize(rng.normal(size=D))


def look(p, k=0.8):
    """One live sighting: true direction plus noise (cos between two ~0.6)."""
    return l2_normalize(p + k * rng.normal(size=D) / np.sqrt(D))


def box(cx, cy=200.0, h=180.0):
    w = 0.4 * h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class Sim:
    def __init__(self, index=None, **kw):
        cal = ScoreCalibrator(path=TMP / "absent.json")
        self.a = SceneAnalytics(calibrator=cal, **kw)
        self.fusion = CrossViewFusion(index=index, cross_view_threshold=0.45,
                                      gallery_threshold=0.65)
        self.embeds = 0
        self.events = []

    def step(self, t, scene):
        by_cam = {}
        for cam in ("CAM_A", "CAM_B"):
            dets = scene.get(cam, [])
            tracks = self.a.track(cam, t, [d[0] for d in dets], [0.9] * len(dets), (640, 360))
            obs = []
            for (b, person), trk in zip(dets, tracks):
                if self.a.should_check_face(trk, t):
                    emb = look(person) if person is not None else None
                    if emb is not None:
                        self.embeds += 1
                    self.a.face_result(trk, t, embedding=emb,
                                       thumb="thumb" if emb is not None else None)
                vec, cached = SceneAnalytics.embedding_for(trk)
                obs.append(ViewObservation(camera=cam, box=b, conf=0.9,
                                           crop=np.zeros((1, 1, 3), np.uint8),
                                           embedding=vec, track=trk, cached=cached))
            by_cam[cam] = obs
        fused = self.fusion.process(by_cam)
        ev = self.a.resolve(t, fused)
        self.events.extend(ev)
        return ev, fused

    def run(self, t0, t1, scene_fn):
        t = t0
        while t < t1 - 1e-9:
            self.step(t, scene_fn(t))
            t += DT
        return t

    def kinds(self, kind):
        return [e for e in self.events if e["kind"] == kind]


# =============================================================================
# 1. A stranger walks from camera A to camera B
# =============================================================================
P1, P2, P3, P4 = new_person(), new_person(), new_person(), new_person()
sim = Sim()

sim.run(0.0, 5.0, lambda t: {"CAM_A": [(box(100 + 60 * t), P1)]})
new = sim.kinds("visitor_new")
check("stranger gets a visitor name", len(new) == 1 and "Visitor #1" in new[0]["text"],
      str([e["text"] for e in new]))

sim.run(5.0, 13.0, lambda t: {})
sim.run(13.0, 16.0, lambda t: {"CAM_B": [(box(500 - 60 * (t - 13)), P1)]})
ret = sim.kinds("visitor_return")
check("same stranger recognised on the other camera",
      len(ret) == 1 and "Visitor #1" in ret[0]["text"] and "CAM_A" in ret[0]["text"],
      str([e["text"] for e in ret]))
check("no second visitor was created for them", sim.a.visitors.created == 1,
      "created %d" % sim.a.visitors.created)

walks = sim.a.journeys.model.summary()
check("walk A->B learned", len(walks) == 1 and walks[0]["from"] == "CAM_A"
      and walks[0]["to"] == "CAM_B" and 7.0 <= walks[0]["median_s"] <= 9.0, str(walks))

state = sim.a.state(16.0)
person = next(p for p in state["people"] if p["name"] == "Visitor #1")
check("journey has both cameras in order",
      [v["camera"] for v in person["visits"]] == ["CAM_A", "CAM_B"],
      str([v["camera"] for v in person["visits"]]))

# =============================================================================
# 2. A prediction that comes true
# =============================================================================
sim.run(16.0, 20.0, lambda t: {})
sim.run(20.0, 25.0, lambda t: {"CAM_A": [(box(100 + 50 * (t - 20)), P2)]})
sim.run(25.0, 29.0, lambda t: {})
pred_events = sim.kinds("prediction")
check("leaving A triggers a prediction for B", len(pred_events) >= 1
      and "CAM_B" in pred_events[-1]["text"], str([e["text"] for e in pred_events]))
person = next(p for p in sim.a.state(29.0)["people"] if p["key"] == "v:2")
check("prediction is pending with an ETA", person["prediction"]
      and person["prediction"]["status"] == "pending" and person["prediction"]["to"] == "CAM_B",
      str(person["prediction"]))

sim.run(32.0, 34.0, lambda t: {"CAM_B": [(box(500), P2)]})
hits = sim.kinds("prediction_hit")
check("arrival inside the window scores a hit", len(hits) == 1, str([e["text"] for e in hits]))
stats = sim.a.journeys.prediction_stats()
check("hit counted", stats["hits"] == 1 and stats["misses"] == 0, str(stats))

# =============================================================================
# 3. A prediction that does not come true
# =============================================================================
sim.run(34.0, 40.0, lambda t: {})
sim.run(40.0, 42.0, lambda t: {"CAM_A": [(box(150), P3)]})
sim.run(42.0, 95.0, lambda t: {})
check("overdue flagged", len(sim.kinds("overdue")) >= 1)
stats = sim.a.journeys.prediction_stats()
check("never arriving counts as a miss", stats["misses"] == 1 and stats["accuracy"] == 0.5,
      str(stats))

# =============================================================================
# 4. Loitering, with and without a face; track-then-recognise
# =============================================================================
before = sim.embeds
sim.run(95.0, 131.0, lambda t: {"CAM_B": [(box(300), P4)], "CAM_A": [(box(400), None)]})
loiter = sim.kinds("loiter")
named = [e for e in loiter if "Visitor" in e["text"]]
anon = [e for e in loiter if e["text"].startswith("Person ")]
check("loitering alert for a recognised person, once", len(named) == 1, str([e["text"] for e in named]))
check("loitering alert for a person whose face is never seen", len(anon) == 1,
      str([e["text"] for e in anon]))
state = sim.a.state(130.9)
p4 = next(p for p in state["people"] if p["in_view"])
check("dwell time reported", p4["dwell_s"] >= 30 and p4["loitering"], "dwell %s" % p4["dwell_s"])

cycles = int(round((131.0 - 95.0) / DT))
p4_embeds = sim.embeds - before
check("settled identity is not re-embedded every cycle",
      p4_embeds <= cycles // 4, "%d embeddings over %d cycles" % (p4_embeds, cycles))
eff = state["efficiency"]
check("efficiency reported", eff["skipped_pct"] is not None and eff["skipped_pct"] > 50, str(eff))
check("only the person with a face has a name",
      all((r["identity"] is None) != r["has_face"]
          for v in state["views"].values() for r in v["tracks"]))

# =============================================================================
# 5. Two names for one person are merged when both cameras see them at once
# =============================================================================
P5 = new_person()
sim2 = Sim()
sim2.run(0.0, 2.0, lambda t: {"CAM_A": [(box(200), P5)]})
sim2.a.apply_settings({"visitor_threshold": 0.99})     # force a second name
sim2.run(10.0, 11.0, lambda t: {"CAM_B": [(box(300), P5)]})
sim2.a.apply_settings({"visitor_threshold": 0.45})
check("setup: the person was given two names", sim2.a.visitors.created == 2)
sim2.run(11.0, 16.0, lambda t: {"CAM_A": [(box(200), P5)], "CAM_B": [(box(300), P5)]})
check("seen together -> the two names merge", len(sim2.kinds("merged")) == 1
      and len(sim2.a.visitors) == 1, str([e["text"] for e in sim2.kinds("merged")]))

# =============================================================================
# 6. A visitor who is then enrolled keeps their journey under their name
# =============================================================================
for p in TMP.glob("idx.*"):
    p.unlink()
index = ThreatIndex(index_path=TMP / "idx.index", labels_path=TMP / "idx.json", threshold=0.65)
P6 = new_person()
sim3 = Sim(index=index)
sim3.run(0.0, 2.0, lambda t: {"CAM_A": [(box(200), P6)]})
check("unknown person starts as a visitor", sim3.a.visitors.created == 1)
for _ in range(3):
    index.add(look(P6, 0.3), "Alice", persist=False)
sim3.run(2.0, 6.0, lambda t: {"CAM_A": [(box(200), P6)]})
ident = sim3.kinds("identified")
check("visitor identified as the enrolled person", len(ident) == 1 and "Alice" in ident[0]["text"],
      str([e["text"] for e in ident]))
keys = list(sim3.a.journeys.journeys)
check("journey continues under the enrolled name", keys == ["id:Alice"], str(keys))
check("visitor record removed", len(sim3.a.visitors) == 0)

# =============================================================================
# 7. Visitors are forgotten after the time limit
# =============================================================================
P7 = new_person()
sim4 = Sim(visitor_ttl_s=20.0)
sim4.run(0.0, 2.0, lambda t: {"CAM_A": [(box(200), P7)]})
sim4.run(2.0, 30.0, lambda t: {})
check("forgotten after the limit", len(sim4.a.visitors) == 0 and not sim4.a.journeys.journeys)
sim4.run(30.0, 32.0, lambda t: {"CAM_A": [(box(200), P7)]})
check("comes back as a new visitor", sim4.a.visitors.created == 2
      and "Visitor #2" in sim4.kinds("visitor_new")[-1]["text"])

# =============================================================================
# 8. Confidence calibration
# =============================================================================
genuine = np.clip(rng.normal(0.62, 0.08, 300), -1, 1)
impostor = np.clip(rng.normal(0.08, 0.07, 900), -1, 1)
a, b = fit_logistic(genuine, impostor)
check("curve increases with score", a > 0, "a=%.2f b=%.2f" % (a, b))
cal = build_calibration(genuine, impostor, n_identities=6, source="synthetic")
c = ScoreCalibrator(path=TMP / "cal.json", data=None)
c.save(cal)
c2 = ScoreCalibrator(path=TMP / "cal.json")
check("calibration survives save/load", c2.calibrated)
p_hi, p_lo, p_mid = c2.probability(0.62), c2.probability(0.08), c2.probability(cal["score_at_50pct"])
check("same-person score -> high %", p_hi > 0.9, "%.3f" % p_hi)
check("different-person score -> low %", p_lo < 0.1, "%.3f" % p_lo)
check("midpoint -> about 50%", abs(p_mid - 0.5) < 0.02, "%.3f" % p_mid)
check("never claims certainty", c2.probability(1.0) <= 0.99 and c2.probability(-1.0) >= 0.01)
check("EER reported and small for well-separated data", cal["eer"] < 0.05, "EER %.3f" % cal["eer"])
check("uncalibrated gives no percentage",
      ScoreCalibrator(path=TMP / "absent.json").probability(0.9) is None)
for bad, why in [((genuine[:5], impostor), "too few same-person pairs"),
                 ((impostor[:50], genuine[:50]), "labels swapped")]:
    try:
        build_calibration(bad[0], bad[1], n_identities=4, source="x")
        check("refuses: " + why, False, "accepted it")
    except ValueError as exc:
        check("refuses: " + why, True, str(exc)[:60])
try:
    build_calibration(genuine, impostor, n_identities=1, source="x")
    check("refuses: one person only", False)
except ValueError:
    check("refuses: one person only", True)

sim5 = Sim()
sim5.a.calibrator = c2
sim5.run(0.0, 3.0, lambda t: {"CAM_A": [(box(200), P1)]})
idents = [r["identity"] for v in sim5.a.state(3.0)["views"].values() for r in v["tracks"]]
check("confidence shown once calibrated",
      any(i and i.get("confidence") is not None for i in idents), str(idents))

# =============================================================================
# 8b. Everyone is tracked; face work is rationed to the people who need it
# =============================================================================
# Regression: capping *detections* at the face budget made people drop in and
# out of the results whenever more were in view, resetting IDs and dwell.
ra = SceneAnalytics(calibrator=ScoreCalibrator(path=TMP / "absent.json"))
crowd = [box(60 + 80 * k, h=120 + 15 * k) for k in range(6)]        # six people
tr = ra.track("CAM_A", 0.0, crowd, [0.9] * 6, (640, 360))
chosen = ra.select_face_checks(tr, 0.0, budget=2)
check("all six people tracked", len(ra.trackers["CAM_A"].tracks) == 6)
check("face work limited to the budget", len(chosen) == 2, str(sorted(chosen)))
check("the nearest (largest) unrecognised people go first", chosen == {4, 5}, str(sorted(chosen)))
for i in chosen:
    ra.face_result(tr[i], 0.0, embedding=new_person())
tr = ra.track("CAM_A", 0.33, crowd, [0.9] * 6, (640, 360))
chosen = ra.select_face_checks(tr, 0.33, budget=2)
check("next cycle serves the people still waiting", chosen == {2, 3}, str(sorted(chosen)))
ids = [t.label for t in tr]
for k in range(10):
    tr = ra.track("CAM_A", 0.66 + k * 0.33, crowd, [0.9] * 6, (640, 360))
    ra.select_face_checks(tr, 0.66 + k * 0.33, budget=2)
check("nobody's ID changes while they stay in view", [t.label for t in tr] == ids)

# =============================================================================
# 9. The state is JSON and the fused-track rows carry the person
# =============================================================================
state = sim.a.state(131.0)
try:
    json.dumps(state)
    ok = True
except (TypeError, ValueError) as exc:
    ok = False
    print("    ", exc)
check("state is JSON-serialisable", ok)
for key in ("views", "people", "transit", "predictions", "events", "calibration",
            "visitors", "efficiency", "loiter_seconds"):
    check("state has '%s'" % key, key in state)
_, fused = sim2.step(16.0, {"CAM_A": [(box(200), P5)], "CAM_B": [(box(300), P5)]})
row = fused_track_payload(fused[0], sim2.a.calibrator)
check("track row names the person", row["person"] and row["person"].startswith("Visitor"), str(row["person"]))
check("track row shows the tracker IDs", all(v["track_id"] for v in row["views"]))

print("\n" + ("ALL PASS" if not failures else "%d FAILURE(S): %s" % (len(failures), failures)))
sys.exit(1 if failures else 0)
