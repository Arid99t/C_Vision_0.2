// Tracking and label stability, on the device.
//
// Port of Tracker / Track / box_gap and the majority-vote confirmation from
// clients/desktop/ar_app.py. A formula keeps its identity (and so its label
// and its 3D molecule) from frame to frame; a label is only shown once
// several recent reads agree, so one flickered misread is simply outvoted.

import { iou } from './detect.js';

export class Track {
  constructor(id, box, now) {
    this.id = id;
    this.box = { ...box };          // latest detection
    this.smooth = { ...box };       // eased box, for steady overlays
    this.formula = null;            // confirmed label (null until stable)
    this.pretty = null;
    this.name = null;
    this.confidence = 0;
    this.reads = [];                // recent {formula, conf} votes
    this.hits = 1;
    this.lastSeen = now;
    this.lastRead = -1e9;           // when recognition last ran on this track
    this.readBusy = false;
  }

  center() {
    return { x: this.smooth.x + this.smooth.w / 2, y: this.smooth.y + this.smooth.h / 2 };
  }
}

export class Tracker {
  /** @param keepMs how long a track survives without a matching detection */
  constructor({ iouThresh = 0.1, keepMs = 450, ease = 0.45 } = {}) {
    this.tracks = [];
    this.iouThresh = iouThresh;
    this.keepMs = keepMs;
    this.ease = ease;
    this.nextId = 0;
  }

  /** Greedy IoU matching of this frame's boxes to the existing tracks. */
  update(boxes, now) {
    const unmatched = new Set(boxes.keys());
    for (const tr of this.tracks) {
      let best = -1, bestIou = this.iouThresh;
      for (const i of unmatched) {
        const v = iou(tr.box, boxes[i]);
        if (v >= bestIou) { best = i; bestIou = v; }
      }
      if (best >= 0) {
        const b = boxes[best];
        tr.box = { x: b.x, y: b.y, w: b.w, h: b.h };
        const e = this.ease;
        tr.smooth.x += (b.x - tr.smooth.x) * e;
        tr.smooth.y += (b.y - tr.smooth.y) * e;
        tr.smooth.w += (b.w - tr.smooth.w) * e;
        tr.smooth.h += (b.h - tr.smooth.h) * e;
        tr.lastSeen = now;
        tr.hits++;
        unmatched.delete(best);
      }
    }
    for (const i of unmatched) this.tracks.push(new Track(this.nextId++, boxes[i], now));
    this.tracks = this.tracks.filter((t) => now - t.lastSeen <= this.keepMs);
    return this.tracks;
  }
}

/** Shortest gap between two axis-aligned boxes (0 if they overlap). */
export function boxGap(a, b) {
  const dx = Math.max(b.x - (a.x + a.w), a.x - (b.x + b.w), 0);
  const dy = Math.max(b.y - (a.y + a.h), a.y - (b.y + b.h), 0);
  return Math.hypot(dx, dy);
}

/**
 * Add one accepted read to a track and (re)decide its label by majority vote
 * over a sliding window of 2*stable-1 reads; `stable` agreeing reads confirm.
 * Returns true when the confirmed formula changed.
 */
export function vote(track, formula, conf, stable = 2) {
  track.reads.push({ formula, conf });
  const keep = 2 * stable - 1;
  if (track.reads.length > keep) track.reads.splice(0, track.reads.length - keep);
  const tally = new Map();
  for (const r of track.reads) {
    const t = tally.get(r.formula) || { n: 0, sum: 0 };
    t.n++; t.sum += r.conf;
    tally.set(r.formula, t);
  }
  let winner = null, top = null;
  for (const [f, t] of tally) {
    if (!top || t.n > top.n || (t.n === top.n && t.sum > top.sum)) { winner = f; top = t; }
  }
  if (top.n >= stable) {
    const changed = track.formula !== winner;
    track.formula = winner;
    track.confidence = top.sum / top.n;
    return changed;
  }
  return false;
}
