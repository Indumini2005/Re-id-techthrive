"""Re-identifying people who were never enrolled.

The enrolled gallery answers "is this someone on the watchlist?". This
answers the question re-identification is actually named for: "have I seen
this person before - on either camera - even though nobody told me who they
are?". Every new face gets a temporary name, ``Visitor #N``, and is
recognised again when it reappears.

Privacy by construction:

* in memory only - nothing is written to disk;
* each visitor is forgotten ``ttl_s`` seconds after their last sighting;
* only face embeddings are kept, plus one small face thumbnail for the
  operator screen, which is dropped with the visitor.

Each visitor holds up to ``max_samples`` recent embeddings, and is compared
by their normalised mean - several looks at a face from different moments
make a steadier reference than any single frame.

The threshold is a live-vs-live comparison, like the cross-view threshold,
so it defaults to the same value. Remember the failure mode measured on
this model: ArcFace maps poor inputs close together, so two bad faces can
look alike. The face quality gate in face_align.py is what keeps those out
of here; do not lower the threshold to compensate for missing faces.
"""

from collections import deque

import numpy as np

import config
from .vectors import l2_normalize


class Visitor:
    def __init__(self, number, embedding, t, thumb=None, max_samples=12):
        self.number = int(number)
        self.samples = deque(maxlen=max_samples)
        self.samples.append(l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1)))
        self.prototype = self.samples[0]
        self.first_seen = float(t)
        self.last_seen = float(t)
        self.sightings = 1
        self.thumb = thumb
        self.thumb_score = -1.0

    @property
    def key(self):
        return "v:%d" % self.number

    @property
    def name(self):
        return "Visitor #%d" % self.number

    def similarity(self, embedding):
        return float(np.dot(self.prototype, l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))))

    def add(self, embedding, t, thumb=None, score=None):
        self.samples.append(l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1)))
        self.prototype = l2_normalize(np.mean(np.stack(self.samples), axis=0))
        self.last_seen = max(self.last_seen, float(t))
        self.sightings += 1
        # Keep the face thumbnail from the most typical-looking sighting.
        if thumb is not None and (self.thumb is None or (score or 0.0) >= self.thumb_score):
            self.thumb = thumb
            self.thumb_score = score or 0.0


class VisitorGallery:
    def __init__(self, threshold=None, margin=None, ttl_s=None, max_samples=12):
        self.threshold = float(threshold if threshold is not None
                               else getattr(config, "VISITOR_THRESHOLD", 0.45))
        self.margin = float(margin if margin is not None
                            else getattr(config, "VISITOR_MARGIN", 0.05))
        self.ttl_s = float(ttl_s if ttl_s is not None
                           else getattr(config, "VISITOR_TTL_S", 600.0))
        self.max_samples = max_samples
        self.visitors = {}            # key -> Visitor
        self._next = 1
        self.created = 0
        self.merged = 0

    def __len__(self):
        return len(self.visitors)

    def get(self, key):
        return self.visitors.get(key)

    def expire(self, t):
        """Forget anyone not seen for ttl_s. Returns the keys dropped."""
        gone = [k for k, v in self.visitors.items() if float(t) - v.last_seen > self.ttl_s]
        for k in gone:
            del self.visitors[k]
        return gone

    def ranked(self, embedding):
        out = [(v, v.similarity(embedding)) for v in self.visitors.values()]
        out.sort(key=lambda p: p[1], reverse=True)
        return out

    def create(self, embedding, t, thumb=None):
        v = Visitor(self._next, embedding, t, thumb, self.max_samples)
        self._next += 1
        self.created += 1
        self.visitors[v.key] = v
        return v

    def merge(self, keep, drop):
        """Fold ``drop`` into ``keep`` - the same person was given two names."""
        if keep is drop or drop.key not in self.visitors:
            return
        for s in drop.samples:
            keep.samples.append(s)
        keep.prototype = l2_normalize(np.mean(np.stack(keep.samples), axis=0))
        keep.first_seen = min(keep.first_seen, drop.first_seen)
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        keep.sightings += drop.sightings
        if keep.thumb is None:
            keep.thumb = drop.thumb
        del self.visitors[drop.key]
        self.merged += 1

    def assign(self, embedding, t, current_key=None, thumb=None):
        """Decide who this face is among the visitors.

        Returns ``(visitor, score, outcome, merged_keys)`` where outcome is
        ``"new"``, ``"same"`` (continuing), ``"returned"`` (back after a gap),
        or ``"switched"`` (the track had been given the wrong visitor), and
        score is the similarity to the chosen visitor (None when new).
        """
        self.expire(t)
        ranked = self.ranked(embedding)
        current = self.visitors.get(current_key) if current_key else None
        merged = []

        if current is not None:
            s_cur = current.similarity(embedding)
            best, s_best = ranked[0] if ranked else (None, -1.0)
            # Stick with the track's visitor unless another is clearly
            # better: one awkward frame must not rename a person.
            if (best is not None and best is not current and s_best >= self.threshold
                    and s_best >= s_cur + self.margin):
                chosen, score, outcome = best, s_best, "switched"
            else:
                chosen, score, outcome = current, s_cur, "same"
        elif ranked and ranked[0][1] >= self.threshold:
            chosen, score = ranked[0]
            gap = float(t) - chosen.last_seen
            outcome = "returned" if gap >= getattr(config, "VISITOR_RETURN_GAP_S", 5.0) else "same"
        else:
            return self.create(embedding, t, thumb), None, "new", merged

        if score >= self.threshold:
            # Anyone else this face also matches, who also matches the chosen
            # visitor, is the same person given two names earlier - say, when
            # their first look at one camera was a poor one. Fold them into
            # the oldest name, so the label people have already seen stays.
            dups = [v for v, s in ranked if v is not chosen and s >= self.threshold
                    and float(np.dot(v.prototype, chosen.prototype)) >= self.threshold]
            if dups:
                keep = min([chosen] + dups, key=lambda v: v.number)
                for v in [chosen] + dups:
                    if v is not keep:
                        merged.append(v.key)
                        self.merge(keep, v)
                chosen = keep
                score = chosen.similarity(embedding)
            # Only learn from faces that actually resemble the reference, so
            # a mismatched frame cannot drag the prototype off.
            chosen.add(embedding, t, thumb, score)
        else:
            chosen.last_seen = max(chosen.last_seen, float(t))
        return chosen, score, outcome, merged

    def remove(self, key):
        self.visitors.pop(key, None)
