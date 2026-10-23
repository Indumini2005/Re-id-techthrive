"""Procedural CCTV scene generator.

Renders a room from a fixed elevated camera, the way a real surveillance
install sees it: perspective floor, textured surfaces, overhead lighting,
people walking at depth-correct scale. Output frames are CLEAN - the
degradation chain in degradations.py produces the noisy half of each pair.

Two uses:

1. **Training data** when you have no footage yet.

       python -m training.synthesize_cctv frames --out data/synth --count 800

2. **Two synchronized camera views of one scene**, so the dual-camera hub
   and the cross-view fusion can be tested without any hardware. The same
   person is visible from both angles at the same instant.

       python -m training.synthesize_cctv video --out data/synth_video --seconds 60

Read this before relying on it: procedural frames do not carry the full
statistics of real photographs. A denoiser trained only on these will look
good on more of the same and will over-smooth real footage. Use it to
validate the pipeline end to end and to warm-start training; fine-tune on
real frames before you trust any number. ``training/make_dataset.py video``
turns your own phone recordings into exactly the right input.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- texture ---------------------------------------------------------------
def fractal_noise(h, w, rng, persistence=0.62, finest=3):
    """Sum-of-octaves value noise with detail down to a few pixels.

    Real surfaces carry detail at every scale, roughly 1/f. Flat procedural
    fills do not, and a denoiser trained on them learns that smooth is
    always correct - precisely the failure to avoid. So the octaves run
    until features are ``finest`` pixels across.

    They stop there rather than at one pixel on purpose: per-pixel white
    noise in a *clean* target would teach the model to reproduce noise,
    which is the opposite of the job. Texture is spatially correlated;
    sensor noise is not.
    """
    total = np.zeros((h, w), np.float32)
    amplitude, max_amplitude, size = 1.0, 0.0, 4

    while size <= max(h, w) // max(finest, 1):
        gh = max(2, int(size * h / max(w, 1)))
        gw = max(2, size)
        layer = rng.random((gh, gw)).astype(np.float32)
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
        total += layer * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        size *= 2

    return total / max(max_amplitude, 1e-6)


# --- camera ----------------------------------------------------------------
class Camera:
    """Pinhole camera on a world with Z up and the floor at Z = 0."""

    def __init__(self, pos, yaw_deg, pitch_deg, fov_deg, width, height):
        self.pos = np.array(pos, dtype=np.float32)
        self.width, self.height = width, height

        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        forward = np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
        ], dtype=np.float32)
        forward /= np.linalg.norm(forward)

        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)

        self.axes = np.stack([right, down, forward])      # world -> camera
        self.f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        self.cx, self.cy = width / 2.0, height / 2.0

    def project(self, points_world):
        """(N,3) world -> ((N,2) pixels, (N,) depth). Depth <= 0 is behind."""
        rel = np.atleast_2d(points_world).astype(np.float32) - self.pos
        cam = rel @ self.axes.T
        depth = cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-5, 1e-5, depth)
        u = self.f * cam[:, 0] / safe + self.cx
        v = self.f * cam[:, 1] / safe + self.cy
        return np.stack([u, v], axis=1), depth

    def floor_map(self):
        """Per-pixel world (x, y) where each ray meets the floor.

        Returns (wx, wy, mask). Raycasting the floor rather than drawing a
        projected grid is what gives the texture correct perspective
        foreshortening, which is a large part of why a CCTV frame looks the
        way it does.
        """
        us, vs = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )
        dirs_cam = np.stack([
            (us - self.cx) / self.f,
            (vs - self.cy) / self.f,
            np.ones_like(us),
        ], axis=-1)
        dirs_world = dirs_cam @ self.axes            # camera -> world

        dz = dirs_world[..., 2]
        mask = dz < -1e-4                            # pointing down at the floor
        t = np.where(mask, -self.pos[2] / np.where(mask, dz, -1.0), 0.0)
        wx = self.pos[0] + t * dirs_world[..., 0]
        wy = self.pos[1] + t * dirs_world[..., 1]

        # Discard anything implausibly far; it aliases into noise.
        mask &= (t > 0) & (t < 60.0)
        return wx, wy, mask


# --- scene -----------------------------------------------------------------
class Person:
    def __init__(self, rng, room):
        self.room = room
        self.pos = np.array([rng.uniform(1, room[0] - 1), rng.uniform(1, room[1] - 1)], np.float32)
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(0.6, 1.6)
        self.vel = np.array([math.cos(angle), math.sin(angle)], np.float32) * speed
        self.height = rng.uniform(1.58, 1.88)
        self.width = rng.uniform(0.38, 0.52)
        tone = rng.integers(45, 150)
        self.color = tuple(int(np.clip(tone + rng.integers(-28, 28), 20, 210)) for _ in range(3))
        self.phase = rng.uniform(0, 2 * math.pi)

    def step(self, dt):
        self.pos += self.vel * dt
        for i in (0, 1):
            if self.pos[i] < 0.6:
                self.pos[i], self.vel[i] = 0.6, abs(self.vel[i])
            elif self.pos[i] > self.room[i] - 0.6:
                self.pos[i], self.vel[i] = self.room[i] - 0.6, -abs(self.vel[i])
        self.phase += dt * 6.0


class Scene:
    """A room, its fixed clutter, and the people moving through it."""

    def __init__(self, rng, room=(12.0, 9.0), n_people=3, night=False):
        self.rng = rng
        self.room = room
        self.night = night
        self.people = [Person(rng, room) for _ in range(n_people)]

        self.tile = rng.uniform(0.5, 1.1)
        self.floor_tone = rng.uniform(0.28, 0.62)
        self.wall_tone = rng.uniform(0.30, 0.58)
        self.tint = rng.uniform(-0.07, 0.07, size=3).astype(np.float32)

        # Overhead lights: (x, y, radius, strength)
        self.lights = [
            (rng.uniform(1, room[0] - 1), rng.uniform(1, room[1] - 1),
             rng.uniform(2.0, 4.5), rng.uniform(0.35, 0.85))
            for _ in range(rng.integers(2, 5))
        ]
        # Wall fixtures: (x, y, width, height, z_bottom, shade)
        self.fixtures = [
            (rng.uniform(0, room[0]), rng.choice([0.0, room[1]]),
             rng.uniform(0.8, 1.6), rng.uniform(1.4, 2.1), 0.0,
             rng.uniform(0.18, 0.5))
            for _ in range(rng.integers(2, 5))
        ]
        self._tex_cache = {}

    def step(self, dt):
        for p in self.people:
            p.step(dt)

    # -- rendering ---------------------------------------------------------
    def _floor_texture(self, shape):
        if shape not in self._tex_cache:
            self._tex_cache[shape] = fractal_noise(shape[0], shape[1], self.rng)
        return self._tex_cache[shape]

    def render(self, camera):
        h, w = camera.height, camera.width
        img = np.zeros((h, w, 3), np.float32)

        # --- walls / backdrop above the horizon
        grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        wall = self.wall_tone * (0.62 + 0.38 * grad)
        wall = wall + (fractal_noise(h, w, self.rng) - 0.5) * 0.22
        # vertical streaking: panel joints, pipework, wear
        streak = fractal_noise(8, w, self.rng, finest=2)[0]
        wall = wall + (streak[None, :] - 0.5) * 0.09
        img[:] = wall[..., None]

        # --- floor by raycast, so the tiling foreshortens correctly
        wx, wy, mask = camera.floor_map()
        tex = self._floor_texture((h, w))

        tiles = (np.floor(wx / self.tile) + np.floor(wy / self.tile)) % 2.0
        grout = ((np.abs((wx / self.tile) % 1.0 - 0.5) > 0.46) |
                 (np.abs((wy / self.tile) % 1.0 - 0.5) > 0.46))

        floor = self.floor_tone * (0.86 + 0.14 * tiles)
        floor = floor * np.where(grout, 0.74, 1.0)
        floor = floor + (tex - 0.5) * 0.30

        # Scuffs and wear, keyed to world position so they stay put as the
        # people move. Without them the floor is a clean procedural grid,
        # which is not what any real surface looks like.
        scuff = fractal_noise(h, w, self.rng, persistence=0.70, finest=2)
        floor = floor * (0.90 + 0.20 * scuff)

        # pools of light from the overhead fixtures
        light = np.full((h, w), 0.30 if self.night else 0.72, np.float32)
        for lx, ly, radius, strength in self.lights:
            d2 = (wx - lx) ** 2 + (wy - ly) ** 2
            light += strength * np.exp(-d2 / (2 * radius ** 2))
        floor = floor * np.clip(light, 0.05, 2.1)

        img[mask] = floor[mask][..., None]

        # --- fixtures on the walls
        for fx, fy, fw, fh, fz, shade in self.fixtures:
            corners = np.array([
                [fx - fw / 2, fy, fz], [fx + fw / 2, fy, fz],
                [fx + fw / 2, fy, fz + fh], [fx - fw / 2, fy, fz + fh],
            ], np.float32)
            pts, depth = camera.project(corners)
            if np.any(depth <= 0.2):
                continue
            poly = pts.astype(np.int32)
            if poly[:, 0].max() < -50 or poly[:, 0].min() > w + 50:
                continue
            cv2.fillConvexPoly(img, poly, (float(shade),) * 3, lineType=cv2.LINE_AA)

        # --- people, far to near so nearer figures occlude
        order = []
        for person in self.people:
            base = np.array([person.pos[0], person.pos[1], 0.0], np.float32)
            _, depth = camera.project(base[None])
            order.append((float(depth[0]), person))
        order.sort(key=lambda t: -t[0])

        for depth, person in order:
            if depth <= 0.6:
                continue
            self._draw_person(img, camera, person)

        # --- global falloff toward the frame edge
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        img *= np.clip(1.06 - 0.3 * r ** 2, 0.25, 1.1)[..., None]
        img += self.tint

        img = np.clip(img, 0, 1)
        return (img * 255.0).astype(np.uint8)

    def _draw_person(self, img, camera, person):
        swing = math.sin(person.phase) * 0.13
        x, y = float(person.pos[0]), float(person.pos[1])

        pts_world = np.array([
            [x, y, 0.0],                      # feet
            [x, y, person.height * 0.52],     # hip
            [x, y, person.height * 0.86],     # shoulder
            [x, y, person.height],            # crown
        ], np.float32)
        pts, depth = camera.project(pts_world)
        if np.any(depth <= 0.6):
            return

        feet, hip, shoulder, crown = pts
        px_height = abs(crown[1] - feet[1])
        if px_height < 6 or px_height > camera.height * 2.5:
            return

        half = max(2.0, px_height * person.width / max(person.height, 1e-3) * 0.5)
        shade = tuple(c / 255.0 for c in person.color)

        # contact shadow, which is most of what sells a figure as grounded
        cv2.ellipse(img, (int(feet[0]), int(feet[1])),
                    (int(half * 1.5), max(2, int(half * 0.45))), 0, 0, 360,
                    (0.0, 0.0, 0.0), -1, cv2.LINE_AA)

        torso = np.array([
            [shoulder[0] - half, shoulder[1]], [shoulder[0] + half, shoulder[1]],
            [hip[0] + half * 0.82, hip[1]],
            [feet[0] + half * 0.55 + swing * half, feet[1]],
            [feet[0] - half * 0.55 - swing * half, feet[1]],
            [hip[0] - half * 0.82, hip[1]],
        ], np.int32)
        cv2.fillConvexPoly(img, torso, shade, lineType=cv2.LINE_AA)

        head_r = max(2, int(px_height * 0.072))
        head_y = int(shoulder[1] - head_r * 1.15)
        cv2.ellipse(img, (int(shoulder[0]), head_y),
                    (head_r, int(head_r * 1.2)), 0, 0, 360,
                    tuple(min(1.0, c * 1.25 + 0.1) for c in shade), -1, cv2.LINE_AA)


# --- camera rigs -----------------------------------------------------------
def build_cameras(room, width, height, rng, count=1):
    """One camera, or two looking at the same room from different corners.

    The two-camera rig is deliberately wide-baseline: both see every subject,
    but from angles far enough apart that cross-view association is a real
    test rather than a near-duplicate comparison.
    """
    rw, rh = room
    if count == 1:
        return [Camera(
            pos=(-1.5, rh / 2, rng.uniform(2.6, 3.4)),
            yaw_deg=0.0, pitch_deg=rng.uniform(18, 30),
            fov_deg=rng.uniform(58, 74), width=width, height=height,
        )]

    return [
        Camera(pos=(-1.5, -1.5, 3.0), yaw_deg=38.0, pitch_deg=24.0,
               fov_deg=68.0, width=width, height=height),
        Camera(pos=(rw + 1.5, rh + 1.5, 3.2), yaw_deg=218.0, pitch_deg=26.0,
               fov_deg=68.0, width=width, height=height),
    ]


# --- commands --------------------------------------------------------------
def cmd_frames(args):
    out = Path(args.out) / "clean"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    written = 0
    scenes = 0
    while written < args.count:
        scenes += 1
        scene = Scene(
            rng,
            room=(rng.uniform(8, 16), rng.uniform(6, 12)),
            n_people=int(rng.integers(args.min_people, args.max_people + 1)),
            night=bool(rng.random() < args.night_fraction),
        )
        camera = build_cameras((scene.room[0], scene.room[1]),
                               args.width, args.height, rng, count=1)[0]

        # A handful of frames per scene: enough variety in people positions
        # without paying to rebuild the room texture every frame.
        for _ in range(min(args.frames_per_scene, args.count - written)):
            scene.step(rng.uniform(0.3, 1.2))
            frame = scene.render(camera)
            cv2.imwrite(str(out / ("%06d.png" % written)), frame)
            written += 1
            if written % 50 == 0:
                print("  %d / %d frames (%d scenes)" % (written, args.count, scenes))

    print("-" * 58)
    print("wrote %d clean frames to %s" % (written, out))
    print("\ntrain with:")
    print("  python -m training.train_denoiser --train_dir %s \\" % out)
    print("      --arch nano --preset cctv --epochs 100 --patch_size 128")


def cmd_video(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scene = Scene(rng, room=(12.0, 9.0),
                  n_people=int(rng.integers(args.min_people, args.max_people + 1)),
                  night=bool(rng.random() < args.night_fraction))
    cameras = build_cameras((12.0, 9.0), args.width, args.height, rng, count=2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    paths = [out / "cam_a.mp4", out / "cam_b.mp4"]
    writers = [cv2.VideoWriter(str(p), fourcc, args.fps, (args.width, args.height))
               for p in paths]
    if not all(wr.isOpened() for wr in writers):
        raise SystemExit("could not open the video writers (missing codec?)")

    total = int(args.seconds * args.fps)
    dt = 1.0 / args.fps
    for i in range(total):
        scene.step(dt)
        for camera, writer in zip(cameras, writers):
            writer.write(scene.render(camera))
        if (i + 1) % args.fps == 0:
            print("  %d / %ds" % ((i + 1) // args.fps, args.seconds))

    for writer in writers:
        writer.release()

    print("-" * 58)
    print("wrote %s and %s" % (paths[0].name, paths[1].name))
    print("\nBoth cameras watch the same room from opposite corners, frame-")
    print("synchronized, so every subject appears in both views at once.")
    print("\ntest the hub with them:")
    print("  python -m hub.server --open")
    print("  python node_agent.py --name CAM_A --source %s --hub http://127.0.0.1:8765" % paths[0])
    print("  python node_agent.py --name CAM_B --source %s --hub http://127.0.0.1:8765" % paths[1])


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic CCTV footage")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--width", type=int, default=960)
        p.add_argument("--height", type=int, default=540)
        p.add_argument("--min_people", type=int, default=1)
        p.add_argument("--max_people", type=int, default=4)
        p.add_argument("--night_fraction", type=float, default=0.3)
        p.add_argument("--seed", type=int, default=0)

    p_f = sub.add_parser("frames", help="clean stills for denoiser training")
    common(p_f)
    p_f.add_argument("--count", type=int, default=800)
    p_f.add_argument("--frames_per_scene", type=int, default=8)
    p_f.set_defaults(func=cmd_frames)

    p_v = sub.add_parser("video", help="two synchronized views, for the hub demo")
    common(p_v)
    p_v.add_argument("--seconds", type=int, default=60)
    p_v.add_argument("--fps", type=int, default=15)
    p_v.set_defaults(func=cmd_video)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
