// 3D chemistry layer (Three.js): ball-and-stick molecules floating over the
// recognised formulas, and the animated reaction when two of them meet.
//
// Everything drawn here comes from static/data/chem3d.json:
//   * a molecule's shape, bonds, charges and electron counts (mol3d.py), and
//   * for a reaction, the full plan from reaction3d.py — which atom becomes
//     which, which bonds break and form, and where every valence electron
//     goes. This file only PLAYS that plan; it invents no chemistry.
//
// The animation is a schematic of what changes (bonds, electrons, oxidation
// states), not a step-by-step reaction mechanism.
//
// Scene units are CSS pixels on the z = 0 plane (y up), so a screen point
// (x, y) is the world point (x, height - y, 0).

import * as THREE from '../vendor/three.module.js';

const FOV = 35;
const TAN = Math.tan((FOV * Math.PI) / 360);
const BOND_R = 0.085;                    // stick radius, angstrom
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const COL = {
  bond: 0xd9dde6, broke: 0xff4a3d, formed: 0x5dff9b,
  lone: 0x6fe3ff, shared: 0xffd84a, hot: 0xff8ae2,
};

const easeInOut = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
const easeOut = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : 1 - Math.pow(1 - t, 3));
const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);

// ---------------------------------------------------------------- textures --
/** An electron dot: a white core (tinted by the material / vertex colour) with a
 *  dark rim, so it stays visible over white paper as well as over a dark desk
 *  (a plain additive glow would vanish on paper). */
let dotTex = null;
function dotTexture() {
  if (dotTex) return dotTex;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, '#ffffff');
  grad.addColorStop(0.46, '#f2f2f2');
  grad.addColorStop(0.56, 'rgba(8,12,26,0.9)');
  grad.addColorStop(0.78, 'rgba(8,12,26,0.35)');
  grad.addColorStop(1, 'rgba(8,12,26,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  dotTex = new THREE.CanvasTexture(c);
  dotTex.colorSpace = THREE.SRGBColorSpace;
  return dotTex;
}

/** Element symbol to print on an atom: dark on light atoms, white on dark ones. */
const letterTex = new Map();
function letterTexture(el, hexColor) {
  if (letterTex.has(el)) return letterTex.get(el);
  const r = parseInt(hexColor.slice(1, 3), 16), gg = parseInt(hexColor.slice(3, 5), 16),
    b = parseInt(hexColor.slice(5, 7), 16);
  const light = 0.299 * r + 0.587 * gg + 0.114 * b > 150;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.font = `700 ${el.length > 1 ? 30 : 38}px system-ui, sans-serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = light ? 'rgba(20,26,40,0.92)' : 'rgba(255,255,255,0.95)';
  g.fillText(el, 32, 35);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  letterTex.set(el, t);
  return t;
}

let glowTex = null;
function glowTexture() {
  if (glowTex) return glowTex;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.85)');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.22)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  glowTex = new THREE.CanvasTexture(c);
  glowTex.colorSpace = THREE.SRGBColorSpace;
  return glowTex;
}

const textTex = new Map();
function chargeText(q) {
  const mag = Math.abs(q) === 1 ? '' : String(Math.abs(q));
  return mag + (q > 0 ? '+' : '−');
}
function textTexture(text, color) {
  const key = text + color;
  if (textTex.has(key)) return textTex.get(key);
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.font = '700 40px system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.lineWidth = 7;
  g.strokeStyle = 'rgba(0,0,0,0.75)';
  g.strokeText(text, 32, 34);
  g.fillStyle = color;
  g.fillText(text, 32, 34);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  textTex.set(key, t);
  return t;
}

// ------------------------------------------------------- electron geometry --
const FIB = (() => {
  const pts = [], ga = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < 48; k++) {
    const y = 1 - (2 * (k + 0.5)) / 48, r = Math.sqrt(Math.max(0, 1 - y * y));
    pts.push([r * Math.cos(ga * k), y, r * Math.sin(ga * k)]);
  }
  return pts;
})();

/** Directions for an atom's lone pairs: as far as possible from its bonds and
 *  from each other. pos: flat xyz array; neighbours: atom indices. */
function loneDirections(a, pos, neighbours, pairs) {
  const taken = [];
  for (const b of neighbours) {
    const dx = pos[3 * b] - pos[3 * a], dy = pos[3 * b + 1] - pos[3 * a + 1], dz = pos[3 * b + 2] - pos[3 * a + 2];
    const n = Math.hypot(dx, dy, dz) || 1;
    taken.push([dx / n, dy / n, dz / n]);
  }
  const out = [];
  for (let p = 0; p < pairs; p++) {
    let best = FIB[(7 * p) % FIB.length], bestScore = -Infinity;
    if (taken.length) {
      for (const c of FIB) {
        let worst = -1;
        for (const t of taken) worst = Math.max(worst, c[0] * t[0] + c[1] * t[1] + c[2] * t[2]);
        if (-worst > bestScore) { bestScore = -worst; best = c; }
      }
    }
    out.push(best);
    taken.push(best);
  }
  return out;
}

/** Where a non-bonding electron sits: atom a, electron index k of its lone set. */
function lonePoint(out, o, pos, a, radius, dirs, k) {
  const d = dirs[k >> 1] || FIB[0];
  // the two electrons of a pair sit side by side, tangent to the atom
  let tx = d[1], ty = -d[0], tz = 0;
  let tn = Math.hypot(tx, ty, tz);
  if (tn < 1e-3) { tx = 1; ty = 0; tz = 0; tn = 1; }
  const side = (k & 1 ? 1 : -1) * 0.13;
  const r = radius + 0.2;
  out[o] = pos[3 * a] + d[0] * r + (tx / tn) * side;
  out[o + 1] = pos[3 * a + 1] + d[1] * r + (ty / tn) * side;
  out[o + 2] = pos[3 * a + 2] + d[2] * r + (tz / tn) * side;
}

/** Where a bonding electron sits: electron k of the `count` shared between a and
 *  b, circling the bond and leaning toward the more electronegative atom. */
function bondPoint(out, o, pos, a, b, k, count, lean, spin) {
  const ax = pos[3 * a], ay = pos[3 * a + 1], az = pos[3 * a + 2];
  let dx = pos[3 * b] - ax, dy = pos[3 * b + 1] - ay, dz = pos[3 * b + 2] - az;
  const len = Math.hypot(dx, dy, dz) || 1;
  dx /= len; dy /= len; dz /= len;
  let ux = -dy, uy = dx, uz = 0;                       // u perpendicular to the bond
  let un = Math.hypot(ux, uy, uz);
  if (un < 1e-3) { ux = 1; uy = 0; uz = 0; un = 1; }
  ux /= un; uy /= un; uz /= un;
  const vx = dy * uz - dz * uy, vy = dz * ux - dx * uz, vz = dx * uy - dy * ux;
  const ang = (2 * Math.PI * k) / Math.max(1, count) + spin;
  const rr = 0.15 + 0.03 * (count > 2 ? 1 : 0);
  const f = (0.5 + lean) * len;
  out[o] = ax + dx * f + (ux * Math.cos(ang) + vx * Math.sin(ang)) * rr;
  out[o + 1] = ay + dy * f + (uy * Math.cos(ang) + vy * Math.sin(ang)) * rr;
  out[o + 2] = az + dz * f + (uz * Math.cos(ang) + vz * Math.sin(ang)) * rr;
}

// ---------------------------------------------------------------- the scene --
export class Scene3D {
  /**
   * @param canvas  WebGL canvas covering the view
   * @param chem    Chem instance (bundle with elements / molecules / reactions)
   * @param labels  absolutely-positioned element that receives text labels
   */
  constructor(canvas, chem, labels) {
    this.chem = chem;
    this.labelLayer = labels;
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x334466, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(-0.5, 0.8, 1);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x9fc4ff, 1.1);
    rim.position.set(0.8, -0.3, -0.6);
    this.scene.add(rim);
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 10, 6000);
    this.sphere = new THREE.SphereGeometry(1, 28, 20);
    this.dot = new THREE.SphereGeometry(1, 8, 6);
    this.stick = new THREE.CylinderGeometry(1, 1, 1, 12, 1, true);
    this.atomMats = new Map();
    this.views = new Map();          // track id -> MoleculeView
    this.player = null;
    this.showElectrons = true;
    this.time = 0;
    this.w = 1; this.h = 1;
  }

  resize(w, h, dpr = Math.min(window.devicePixelRatio || 1, 2)) {
    this.w = w; this.h = h;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.position.set(w / 2, h / 2, h / 2 / TAN);
    this.camera.lookAt(w / 2, h / 2, 0);
    this.camera.updateProjectionMatrix();
  }

  atomMaterial(el) {
    let m = this.atomMats.get(el);
    if (!m) {
      const info = this.chem.bundle.elements[el] || { color: '#c8c8c8' };
      m = new THREE.MeshStandardMaterial({ color: info.color, roughness: 0.3, metalness: 0.08 });
      this.atomMats.set(el, m);
    }
    return m;
  }

  en(el) {
    return (this.chem.bundle.elements[el] || { en: 1.8 }).en;
  }

  /** How far the shared electrons of a-b lean toward the more electronegative atom. */
  lean(elA, elB) {
    const d = (this.en(elB) - this.en(elA)) / 1.2;
    return 0.16 * (d < -1 ? -1 : d > 1 ? 1 : d);
  }

  pointSize(px) {
    return px / TAN;       // PointsMaterial size that shows as `px` CSS pixels on z = 0
  }

  /** A set of electron dots. With `color` null each dot carries its own colour. */
  makePoints(capacity, color, px) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
    if (color === null) {
      geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(capacity * 3).fill(1), 3));
    }
    geo.setDrawRange(0, 0);
    const mat = new THREE.PointsMaterial({
      size: this.pointSize(px), map: dotTexture(), transparent: true,
      depthWrite: false, sizeAttenuation: true, vertexColors: color === null,
    });
    if (color !== null) mat.color.setHex(color);
    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    return pts;
  }

  /** Sprite with the element's symbol, to sit on the camera-facing side of an atom. */
  letter(el) {
    const info = this.chem.bundle.elements[el] || { color: '#c8c8c8' };
    return new THREE.Sprite(new THREE.SpriteMaterial({
      map: letterTexture(el, info.color), transparent: true, depthWrite: false }));
  }

  label(cls) {
    const el = document.createElement('div');
    el.className = 'lbl ' + cls;
    this.labelLayer.appendChild(el);
    return el;
  }

  /** CSS-pixel position of a world point (already in world space). */
  toScreen(v) {
    v.project(this.camera);
    return { x: (v.x * 0.5 + 0.5) * this.w, y: (-v.y * 0.5 + 0.5) * this.h };
  }

  /**
   * Keep one floating molecule per confirmed formula.
   * @param list [{id, formula, template, x, y, size, hidden}] — (x, y) is the
   *   CSS-pixel point the molecule floats at, size the height of its card text.
   */
  syncMolecules(list) {
    const seen = new Set();
    for (const it of list) {
      seen.add(it.id);
      let view = this.views.get(it.id);
      if (view && view.formula !== it.formula) { view.dispose(); view = null; }
      if (!view) {
        view = new MoleculeView(this, it.formula, it.template, it.composition);
        view.jump(it.x, this.h - it.y);
        this.views.set(it.id, view);
      }
      view.setTarget(it.x, this.h - it.y, it.size, !!it.hidden);
    }
    for (const [id, view] of this.views) {
      if (!seen.has(id)) { view.dispose(); this.views.delete(id); }
    }
  }

  /** World position of the molecule shown for a formula (for the reaction to start from). */
  anchorOf(formula) {
    for (const view of this.views.values()) {
      if (view.formula === formula) return view.group.position.clone();
    }
    return null;
  }

  /** Start animating `reaction` (an entry of the bundle that has a .plan). */
  startReaction(reaction, centre) {
    this.endReaction();
    const anchors = {};
    for (const [f] of reaction.plan.reactants) anchors[f] = this.anchorOf(f);
    this.player = new ReactionPlayer(this, reaction, anchors, centre);
    return this.player;
  }

  endReaction() {
    if (this.player) { this.player.dispose(); this.player = null; }
  }

  update(dt) {
    this.time += dt;
    for (const view of this.views.values()) view.update(dt);
    if (this.player) this.player.update(dt);
    this.renderer.render(this.scene, this.camera);
  }
}

// ----------------------------------------------------- one floating molecule --
class MoleculeView {
  constructor(s3, formula, template, composition) {
    this.s3 = s3;
    this.formula = formula;
    this.template = template || fallbackTemplate(composition || {}, s3.chem);
    const t = this.template;
    this.group = new THREE.Group();
    this.inner = new THREE.Group();
    this.group.add(this.inner);
    s3.scene.add(this.group);
    this.target = new THREE.Vector3();
    this.scale = 1; this.targetScale = 1;
    this.alpha = 0; this.hidden = false;
    this.spin = Math.random() * 6.28;

    const n = t.atoms.length;
    this.pos = new Float32Array(n * 3);
    this.radii = new Float32Array(n);
    this.mats = [];
    this.letters = [];
    const fade = (m) => { const c = m.clone(); c.transparent = true; this.mats.push(c); return c; };
    t.atoms.forEach((a, i) => {
      this.pos.set([a[1], a[2], a[3]], 3 * i);
      this.radii[i] = a[5];
      const mesh = new THREE.Mesh(s3.sphere, fade(s3.atomMaterial(a[0])));
      mesh.position.set(a[1], a[2], a[3]);
      mesh.scale.setScalar(a[5]);
      this.inner.add(mesh);
      if (n <= 14) {                                   // element symbols, while there is room
        const sp = s3.letter(a[0]);
        sp.scale.setScalar(a[5] * 1.25);
        this.mats.push(sp.material);
        this.inner.add(sp);
        this.letters.push({ sp, i, r: a[5] });
      }
      if (a[4]) {
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({
          map: textTexture(chargeText(a[4]), a[4] > 0 ? '#ffd1c4' : '#c4e6ff'),
          transparent: true, depthTest: false }));
        sp.position.set(a[1], a[2] + a[5] + 0.32, a[3]);
        sp.scale.setScalar(0.62);
        this.mats.push(sp.material);
        this.inner.add(sp);
      }
    });
    const bondMat = fade(new THREE.MeshStandardMaterial({ color: COL.bond, roughness: 0.45, metalness: 0.1 }));
    const va = new THREE.Vector3(), vb = new THREE.Vector3(), dir = new THREE.Vector3(), off = new THREE.Vector3();
    for (const [i, j, order] of t.bonds) {
      va.fromArray(this.pos, 3 * i); vb.fromArray(this.pos, 3 * j);
      dir.subVectors(vb, va);
      const len = dir.length();
      dir.normalize();
      off.set(-dir.y, dir.x, 0);
      if (off.lengthSq() < 1e-6) off.set(1, 0, 0);
      off.normalize();
      for (let k = 0; k < order; k++) {
        const m = new THREE.Mesh(s3.stick, bondMat);
        const shift = (k - (order - 1) / 2) * 0.2;
        m.position.copy(va).add(vb).multiplyScalar(0.5).addScaledVector(off, shift);
        m.quaternion.setFromUnitVectors(Y_AXIS, dir);
        m.scale.set(order > 1 ? BOND_R * 0.72 : BOND_R, len, order > 1 ? BOND_R * 0.72 : BOND_R);
        this.inner.add(m);
      }
    }
    const linkMat = fade(new THREE.MeshBasicMaterial({ color: 0x8fd8ff, opacity: 0.7 }));
    for (const [i, j] of t.ionic || []) {             // electrostatic contact: dotted link
      va.fromArray(this.pos, 3 * i); vb.fromArray(this.pos, 3 * j);
      for (let k = 1; k <= 5; k++) {
        const d = new THREE.Mesh(s3.dot, linkMat);
        d.position.lerpVectors(va, vb, k / 6);
        d.scale.setScalar(0.045);
        this.inner.add(d);
      }
    }

    // valence electrons: lone ones on atoms, shared ones circling each bond
    const nb = Array.from({ length: n }, () => []);
    for (const [i, j] of t.bonds) { nb[i].push(j); nb[j].push(i); }
    this.loneDirs = t.atoms.map((_, i) => loneDirections(i, this.pos, nb[i], Math.ceil((t.lone?.[i] || 0) / 2)));
    this.nLone = (t.lone || []).reduce((a, b) => a + b, 0);
    this.nShared = t.bonds.reduce((a, b) => a + 2 * b[2], 0);
    this.lonePts = s3.makePoints(Math.max(1, this.nLone), COL.lone, 9);
    this.sharedPts = s3.makePoints(Math.max(1, this.nShared), COL.shared, 9);
    this.inner.add(this.lonePts, this.sharedPts);
    this.mats.push(this.lonePts.material, this.sharedPts.material);
    const lp = this.lonePts.geometry.attributes.position.array;
    let o = 0;
    t.atoms.forEach((a, i) => {
      for (let k = 0; k < (t.lone?.[i] || 0); k++, o += 3) lonePoint(lp, o, this.pos, i, a[5], this.loneDirs[i], k);
    });
    this.lonePts.geometry.setDrawRange(0, this.nLone);
    this.lonePts.geometry.attributes.position.needsUpdate = true;

    this.el = s3.label('mol');
    this.el.innerHTML = `<b>${pretty(formula)}</b>${t.name ? `<span>${t.name}</span>` : ''}`
      + (t.fallback ? '<span class="dim">atoms only — structure not in library</span>' : '');
    this.radius = t.radius;
  }

  jump(x, y) {
    this.group.position.set(x, y, 0);
    this.target.set(x, y, 0);
  }

  setTarget(x, y, cardH, hidden) {
    // floating size: follows the size of the writing, within sensible limits
    const want = Math.min(Math.max(cardH * 1.15, 44), 92);
    this.targetScale = Math.min(want / Math.max(0.8, this.radius), 60);
    this.shown = Math.max(0.8, this.radius) * this.targetScale;      // on-screen radius
    this.target.set(x, y + this.shown + 20, 40);  // float above the card, slightly toward the viewer
    this.hidden = hidden;
  }

  update(dt) {
    const s3 = this.s3;
    const k = Math.min(1, dt * 7);
    this.group.position.lerp(this.target, k);
    this.scale += (this.targetScale - this.scale) * k;
    this.alpha += ((this.hidden ? 0 : 1) - this.alpha) * Math.min(1, dt * 5);
    this.group.visible = this.alpha > 0.02;
    this.inner.scale.setScalar(this.scale * (0.6 + 0.4 * this.alpha));
    this.spin += dt * 0.55;
    this.inner.rotation.set(0.3, this.spin, 0);
    if (this.letters.length) {
      // keep each symbol on the side of its atom that faces the viewer
      const toCam = new THREE.Vector3(0, 0, 1).applyQuaternion(this.inner.quaternion.clone().invert());
      for (const { sp, i, r } of this.letters) {
        sp.position.set(this.pos[3 * i] + toCam.x * r * 1.04, this.pos[3 * i + 1] + toCam.y * r * 1.04,
          this.pos[3 * i + 2] + toCam.z * r * 1.04);
      }
    }
    for (const m of this.mats) m.opacity = (m.userData.base ?? (m.userData.base = m.opacity)) * this.alpha;
    this.lonePts.visible = this.sharedPts.visible = s3.showElectrons;
    if (s3.showElectrons && this.nShared) {
      const sp = this.sharedPts.geometry.attributes.position.array;
      const t = this.template;
      let o = 0;
      for (const [i, j, order] of t.bonds) {
        const lean = s3.lean(t.atoms[i][0], t.atoms[j][0]);
        for (let e = 0; e < 2 * order; e++, o += 3) {
          bondPoint(sp, o, this.pos, i, j, e, 2 * order, lean, s3.time * 2.2);
        }
      }
      this.sharedPts.geometry.setDrawRange(0, this.nShared);
      this.sharedPts.geometry.attributes.position.needsUpdate = true;
    }
    // name tag ABOVE the molecule, so it never covers the handwriting below it
    const p = s3.toScreen(this.group.position.clone().add(new THREE.Vector3(0, this.radius * this.scale + 8, 0)));
    this.el.style.transform = `translate(-50%, -100%) translate(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px)`;
    this.el.style.opacity = this.alpha.toFixed(2);
  }

  dispose() {
    this.s3.scene.remove(this.group);
    for (const m of this.mats) m.dispose();
    this.lonePts.geometry.dispose();
    this.sharedPts.geometry.dispose();
    this.el.remove();
  }
}

/** For a valid formula with no structure in the library: just its atoms, in a
 *  loose cluster, with no bonds claimed. */
function fallbackTemplate(composition, chem) {
  const atoms = [];
  const els = Object.keys(composition);
  let total = 0;
  for (const el of els) total += Math.min(composition[el], 40);
  let k = 0;
  const spread = 0.75 * Math.cbrt(Math.max(1, total)) + 0.2;
  for (const el of els) {
    for (let c = 0; c < Math.min(composition[el], 40); c++, k++) {
      const p = FIB[(k * 5) % FIB.length];
      const r = total === 1 ? 0 : spread * (0.45 + 0.55 * ((k * 0.618) % 1));
      atoms.push([el, p[0] * r, p[1] * r, p[2] * r, 0, el === 'H' ? 0.3 : 0.42]);
    }
  }
  return { atoms, bonds: [], ionic: [], lone: atoms.map(() => 0), radius: spread + 0.5, name: null, fallback: true };
}

export function pretty(formula) {
  return formula.replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[+d]);
}

// ------------------------------------------------------- reaction animation --
class ReactionPlayer {
  constructor(s3, reaction, anchors, centre) {
    this.s3 = s3;
    this.reaction = reaction;
    const plan = this.plan = reaction.plan;
    const n = this.n = plan.atoms.length;
    this.t = 0;
    this.speed = 1;

    // timeline (seconds): gather -> break -> move -> form -> done
    const move = Math.min(4.6, 2.4 + 0.02 * n);
    this.T = { gather: 1.5, brk: 3.2, move: 3.2 + move, form: 3.2 + move + 1.5 };
    this.T.end = this.T.form + 0.8;

    // arena: plan coordinates are angstrom; fit the plan's box on screen
    const [lo, hi] = plan.box;
    const bw = hi[0] - lo[0], bh = hi[1] - lo[1];
    this.scale = Math.min((s3.w * 0.9) / bw, (s3.h * 0.4) / bh, 46);
    this.mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    this.group = new THREE.Group();
    const c = centre || { x: s3.w / 2, y: s3.h * 0.46 };
    const half = (bw * this.scale) / 2;
    const cx = Math.min(Math.max(c.x, half + 6), Math.max(half + 6, s3.w - half - 6));
    this.group.position.set(cx, s3.h - Math.min(Math.max(c.y, s3.h * 0.3), s3.h * 0.56), 60);
    this.group.scale.setScalar(this.scale);
    s3.scene.add(this.group);

    // atoms
    this.start = new Float32Array(n * 3);
    this.end = new Float32Array(n * 3);
    this.from = new Float32Array(n * 3);          // where each atom flies in from
    this.pos = new Float32Array(n * 3);           // current, arena coordinates
    this.rad = new Float32Array(n);
    this.el = [];
    this.meshes = [];
    this.charge = [];
    this.arc = new Float32Array(n * 3);
    this.jit = new Float32Array(n * 3);
    const toLocal = (v) => [(v.x - this.group.position.x) / this.scale,
      (v.y - this.group.position.y) / this.scale, (v.z - this.group.position.z) / this.scale];
    plan.atoms.forEach((a, i) => {
      for (let k = 0; k < 3; k++) {
        this.start[3 * i + k] = a[1 + k] - this.mid[k];
        this.end[3 * i + k] = a[4 + k] - this.mid[k];
      }
      this.el.push(a[0]);
      const mesh = new THREE.Mesh(s3.sphere, s3.atomMaterial(a[0]));
      this.group.add(mesh);
      this.meshes.push(mesh);
      const sprites = [a[7], a[8]].map((q) => {
        if (!q) return null;
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({
          map: textTexture(chargeText(q), q > 0 ? '#ffd1c4' : '#c4e6ff'),
          transparent: true, depthTest: false, opacity: 0 }));
        sp.scale.setScalar(0.62);
        this.group.add(sp);
        return sp;
      });
      this.charge.push(sprites);
      // a sideways bulge so atoms swing past each other instead of colliding
      const dx = this.end[3 * i] - this.start[3 * i], dy = this.end[3 * i + 1] - this.start[3 * i + 1];
      const d = Math.hypot(dx, dy);
      const sgn = i % 2 ? 1 : -1;
      this.arc[3 * i] = d > 1e-3 ? (-dy / d) * 0.22 * d * sgn : 0;
      this.arc[3 * i + 1] = d > 1e-3 ? (dx / d) * 0.22 * d * sgn : 0;
      this.arc[3 * i + 2] = 0.6 * sgn + 0.25 * Math.sin(i * 2.4);
      for (let k = 0; k < 3; k++) this.jit[3 * i + k] = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453 % 1;
    });
    // fly-in origin: the molecule floating over that reactant's card (if tracked)
    const rmol = plan.atoms.map((a) => a[11]);
    plan.r_mols.forEach((m, mi) => {
      const anchor = anchors[m.f] ? toLocal(anchors[m.f]) : null;
      for (let i = 0; i < n; i++) {
        if (rmol[i] !== mi) continue;
        for (let k = 0; k < 3; k++) {
          const rel = this.start[3 * i + k] - (m.c[k] - this.mid[k]);
          this.from[3 * i + k] = anchor ? anchor[k] + rel * 0.55 : this.start[3 * i + k] + (k === 1 ? 4 : 0);
        }
      }
    });

    // bonds: before / after orders for every bonded pair on either side
    const pair = new Map();
    for (const [a, b, o] of plan.bonds_r) pair.set(a + '-' + b, { a, b, r: o, p: 0 });
    for (const [a, b, o] of plan.bonds_p) {
      const e = pair.get(a + '-' + b);
      if (e) e.p = o; else pair.set(a + '-' + b, { a, b, r: 0, p: o });
    }
    this.bonds = [...pair.values()];
    for (const bd of this.bonds) {
      bd.lean = s3.lean(this.el[bd.a], this.el[bd.b]);
      bd.mat = new THREE.MeshStandardMaterial({ color: COL.bond, roughness: 0.45, metalness: 0.1,
        transparent: true, emissive: 0x000000 });
      bd.sticks = [];
      for (let k = 0; k < Math.max(bd.r, bd.p); k++) {
        const m = new THREE.Mesh(s3.stick, bd.mat);
        this.group.add(m);
        bd.sticks.push(m);
      }
      if (bd.r === 0) {                              // a spark where a new bond snaps shut
        bd.flash = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: COL.formed,
          transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
        this.group.add(bd.flash);
      }
    }
    // ionic contacts (dotted links), before and after
    const linkMat = () => new THREE.MeshBasicMaterial({ color: 0x8fd8ff, transparent: true, opacity: 0 });
    this.linksR = { mat: linkMat(), items: plan.ionic_r.map((p) => this.makeLink(p)) };
    this.linksP = { mat: linkMat(), items: plan.ionic_p.map((p) => this.makeLink(p)) };
    for (const l of this.linksR.items) for (const d of l.dots) d.material = this.linksR.mat;
    for (const l of this.linksP.items) for (const d of l.dots) d.material = this.linksP.mat;

    // Electrons. Each dot is coloured by what it currently is — shared in a bond
    // (yellow) or a lone electron on an atom (cyan) — and blends as it changes
    // role. Electrons handed from one atom to another form their own, larger,
    // pink set so the transfer stands out.
    this.e = plan.electrons;
    let ordinary = 0, handed = 0;
    for (const m of this.e) { if (m[8] === 2) handed++; else ordinary++; }
    this.pts = [s3.makePoints(Math.max(1, ordinary), null, 9), s3.makePoints(Math.max(1, handed), COL.hot, 13)];
    this.group.add(...this.pts);
    this.kindColor = [new THREE.Color(COL.lone), new THREE.Color(COL.shared)];
    this.letters = [];
    if (n <= 40) {
      plan.atoms.forEach((a, i) => {
        const sp = s3.letter(a[0]);
        this.group.add(sp);
        this.letters.push(sp);
      });
    }
    this.loneR = new Int32Array(n); this.loneP = new Int32Array(n);
    for (const m of this.e) {
      if (m[0] === 0) this.loneR[m[1]] = Math.max(this.loneR[m[1]], m[3] + 1);
      if (m[4] === 0) this.loneP[m[5]] = Math.max(this.loneP[m[5]], m[7] + 1);
    }
    const nbR = Array.from({ length: n }, () => []), nbP = Array.from({ length: n }, () => []);
    this.orderR = new Map(); this.orderP = new Map();
    for (const [a, b, o] of plan.bonds_r) { nbR[a].push(b); nbR[b].push(a); this.orderR.set(a + '-' + b, o); }
    for (const [a, b, o] of plan.bonds_p) { nbP[a].push(b); nbP[b].push(a); this.orderP.set(a + '-' + b, o); }
    this.dirR = []; this.dirP = [];
    for (let i = 0; i < n; i++) {
      this.dirR.push(loneDirections(i, this.start, nbR[i], Math.ceil(this.loneR[i] / 2)));
      this.dirP.push(loneDirections(i, this.end, nbP[i], Math.ceil(this.loneP[i] / 2)));
    }
    this.tmpA = new Float32Array(3); this.tmpB = new Float32Array(3);

    // heat: a soft burst when the products form
    const dH = reaction.dH;
    this.burst = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture(), color: dH == null ? 0xffffff : (dH < 0 ? 0xff9a3d : 0x5db8ff),
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
    this.group.add(this.burst);
    this.extent = Math.max(bw, bh);

    // text labels for the molecules on each side
    this.labelsR = plan.r_mols.map((m) => this.makeLabel(m));
    this.labelsP = plan.p_mols.map((m) => this.makeLabel(m));
    this.update(0);
  }

  makeLink([a, b]) {
    const dots = [];
    for (let k = 0; k < 5; k++) {
      const d = new THREE.Mesh(this.s3.dot);
      d.scale.setScalar(0.045);
      this.group.add(d);
      dots.push(d);
    }
    return { a, b, dots };
  }

  makeLabel(m) {
    const el = this.s3.label('rx');
    el.textContent = pretty(m.f);
    const radius = this.s3.chem.molecules[m.f]?.radius || 1;
    return { el, c: [m.c[0] - this.mid[0], m.c[1] - this.mid[1] - radius - 0.25, m.c[2] - this.mid[2]] };
  }

  /** 'gather' | 'break' | 'move' | 'form' | 'done' and the caption for it. */
  get phase() {
    const { t, T } = this;
    if (t < T.gather) return 'gather';
    if (t < T.brk) return 'break';
    if (t < T.move) return 'move';
    if (t < T.form) return 'form';
    return 'done';
  }

  get caption() {
    const ph = this.phase;
    if (ph === 'gather') return 'The reactants meet';
    const step = this.plan.steps.find((s) => s.phase === ph);
    return step ? step.text : '';
  }

  get progress() {
    return clamp01(this.t / this.T.end);
  }

  restart() {
    this.t = this.T.gather * 0.6;
  }

  slot(out, pos, kind, a, b, k, after, spin) {
    if (kind === 0) {
      lonePoint(out, 0, pos, a, this.rad[a], (after ? this.dirP : this.dirR)[a], k);
    } else {
      const count = 2 * ((after ? this.orderP : this.orderR).get(a + '-' + b) || 1);
      bondPoint(out, 0, pos, a, b, k, count, this.s3.lean(this.el[a], this.el[b]), spin);
    }
  }

  update(dt) {
    const s3 = this.s3, { T, n, plan } = this;
    this.t += dt * this.speed;
    const t = this.t;
    const uIn = easeOut(t / T.gather);
    const uBreak = clamp01((t - T.gather) / (T.brk - T.gather));
    const uMove = easeInOut((t - T.brk) / (T.move - T.brk));
    const uForm = clamp01((t - T.move) / (T.form - T.move));
    const done = t >= T.form;
    const shake = 0.07 * Math.sin(Math.PI * clamp01((t - T.gather) / (T.move - T.gather)));
    const bulge = Math.sin(Math.PI * uMove);

    // atoms
    for (let i = 0; i < n; i++) {
      const a = plan.atoms[i];
      for (let k = 0; k < 3; k++) {
        const j = 3 * i + k;
        const moving = this.start[j] + (this.end[j] - this.start[j]) * uMove + this.arc[j] * bulge
          + shake * Math.sin(t * 31 + this.jit[j] * 40);
        this.pos[j] = t < T.gather ? this.from[j] + (this.start[j] - this.from[j]) * uIn : moving;
      }
      this.rad[i] = a[9] + (a[10] - a[9]) * uMove;
      const mesh = this.meshes[i];
      mesh.position.fromArray(this.pos, 3 * i);
      mesh.scale.setScalar(this.rad[i] * (t < T.gather ? 0.55 + 0.45 * uIn : 1));
      const [before, after] = this.charge[i];
      if (before) {
        before.position.set(this.pos[3 * i], this.pos[3 * i + 1] + this.rad[i] + 0.32, this.pos[3 * i + 2]);
        before.material.opacity = uIn * (1 - clamp01(uMove * 2.2));
      }
      if (after) {
        after.position.set(this.pos[3 * i], this.pos[3 * i + 1] + this.rad[i] + 0.32, this.pos[3 * i + 2]);
        after.material.opacity = clamp01(uMove * 2.2 - 1.1);
      }
    }

    // bonds
    const va = new THREE.Vector3(), vb = new THREE.Vector3(), dir = new THREE.Vector3(), off = new THREE.Vector3();
    for (const bd of this.bonds) {
      va.fromArray(this.pos, 3 * bd.a); vb.fromArray(this.pos, 3 * bd.b);
      dir.subVectors(vb, va);
      const len = dir.length();
      dir.normalize();
      off.set(-dir.y, dir.x, 0);
      if (off.lengthSq() < 1e-6) off.set(1, 0, 0);
      off.normalize();
      let order, alpha = 1, glow = 0, colour = COL.bond;
      if (bd.p === 0) {                                  // breaks
        order = bd.r;
        alpha = uIn * (1 - easeInOut(uBreak * 1.15));
        glow = Math.sin(Math.PI * clamp01(uBreak * 1.3));
        colour = COL.broke;
      } else if (bd.r === 0) {                           // forms
        order = bd.p;
        alpha = easeOut(uForm * 1.6);
        glow = done ? 0 : 1 - uForm;
        colour = COL.formed;
      } else {                                           // survives (maybe changing order)
        order = t < T.move ? bd.r : bd.p;
        alpha = t < T.gather ? uIn : 1;
        if (bd.r !== bd.p) { glow = Math.sin(Math.PI * uForm); colour = COL.formed; }
      }
      bd.mat.opacity = alpha;
      bd.mat.emissive.setHex(colour).multiplyScalar(glow * 0.9);
      bd.mat.color.setHex(COL.bond).lerp(new THREE.Color(colour), glow * 0.6);
      const thin = order > 1 ? BOND_R * 0.72 : BOND_R;
      bd.sticks.forEach((m, k) => {
        m.visible = k < order && alpha > 0.02 && len < 3.6;
        if (!m.visible) return;
        m.position.copy(va).add(vb).multiplyScalar(0.5).addScaledVector(off, (k - (order - 1) / 2) * 0.2);
        m.quaternion.setFromUnitVectors(Y_AXIS, dir);
        m.scale.set(thin, len, thin);
      });
      if (bd.flash) {
        const f = Math.sin(Math.PI * clamp01(uForm * 1.4));
        bd.flash.position.copy(va).add(vb).multiplyScalar(0.5);
        bd.flash.scale.setScalar(0.5 + 1.6 * f);
        bd.flash.material.opacity = t >= T.move ? 0.85 * f : 0;
      }
    }
    this.linksR.mat.opacity = 0.7 * uIn * (1 - clamp01(uBreak * 1.5));
    this.linksP.mat.opacity = 0.7 * easeOut(uForm);
    for (const set of [this.linksR, this.linksP]) {
      for (const l of set.items) {
        va.fromArray(this.pos, 3 * l.a); vb.fromArray(this.pos, 3 * l.b);
        l.dots.forEach((d, k) => { d.position.lerpVectors(va, vb, (k + 1) / 6); d.visible = set.mat.opacity > 0.02; });
      }
    }

    // electrons: each rides from its place before to its place after
    const show = s3.showElectrons;
    const uE = easeInOut((t - (T.gather + 0.35 * (T.brk - T.gather))) / (T.move + 0.45 * (T.form - T.move) - T.gather - 0.35 * (T.brk - T.gather)));
    const spin = t * 2.2;
    const used = [0, 0];
    const A = this.tmpA, B = this.tmpB;
    const colours = this.pts[0].geometry.attributes.color.array;
    for (const m of this.e) {
      const flag = m[8], set = flag === 2 ? 1 : 0;
      const arr = this.pts[set].geometry.attributes.position.array;
      const o = 3 * used[set]++;
      this.slot(A, this.pos, m[0], m[1], m[2], m[3], false, spin);
      this.slot(B, this.pos, m[4], m[5], m[6], m[7], true, spin);
      const w = flag === 0 ? uMove : uE;
      arr[o] = A[0] + (B[0] - A[0]) * w;
      arr[o + 1] = A[1] + (B[1] - A[1]) * w;
      arr[o + 2] = A[2] + (B[2] - A[2]) * w + (flag === 2 ? 1.1 * Math.sin(Math.PI * w) : 0);
      if (set === 0) {
        const c0 = this.kindColor[m[0]], c1 = this.kindColor[m[4]];
        colours[o] = c0.r + (c1.r - c0.r) * w;
        colours[o + 1] = c0.g + (c1.g - c0.g) * w;
        colours[o + 2] = c0.b + (c1.b - c0.b) * w;
      }
    }
    this.pts.forEach((p, k) => {
      p.visible = show;
      p.geometry.setDrawRange(0, used[k]);
      p.geometry.attributes.position.needsUpdate = true;
    });
    this.pts[0].geometry.attributes.color.needsUpdate = true;
    this.pts[0].material.opacity = uIn;
    // transferred electrons swell while they are in flight
    const travelling = Math.sin(Math.PI * uE);
    this.pts[1].material.size = s3.pointSize(13 + 8 * travelling);
    this.pts[1].material.opacity = uIn;

    if (this.letters.length) {
      // element symbols ride on the viewer-facing side of their atoms
      const toCam = new THREE.Vector3(0, 0, 1).applyQuaternion(this.group.quaternion.clone().invert());
      this.letters.forEach((sp, i) => {
        const r = this.rad[i] * 1.04;
        sp.position.set(this.pos[3 * i] + toCam.x * r, this.pos[3 * i + 1] + toCam.y * r,
          this.pos[3 * i + 2] + toCam.z * r);
        sp.scale.setScalar(this.rad[i] * 1.25);
        sp.material.opacity = uIn;
      });
    }

    // heat burst as the products form
    const b = Math.sin(Math.PI * clamp01((t - T.move) / (T.end - T.move)));
    this.burst.scale.setScalar(this.extent * (0.5 + 1.1 * clamp01((t - T.move) / (T.end - T.move))));
    this.burst.material.opacity = this.reaction.dH == null ? 0 : 0.34 * b;

    // once finished, rock the whole arena gently so the 3D shapes read
    this.group.rotation.y = done ? 0.42 * Math.sin((t - T.form) * 0.55) : 0;
    this.group.rotation.x = done ? 0.1 * Math.sin((t - T.form) * 0.37) : 0;

    // labels: reactant names until things start moving, product names at the end
    const lr = uIn * (1 - clamp01(uBreak * 3)), lp = clamp01((t - T.move) / 0.8);
    const v = new THREE.Vector3();
    for (const [set, alpha] of [[this.labelsR, lr], [this.labelsP, lp]]) {
      for (const l of set) {
        v.set(l.c[0], l.c[1], l.c[2]);
        const p = s3.toScreen(this.group.localToWorld(v));
        l.el.style.transform = `translate(-50%, 0) translate(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px)`;
        l.el.style.opacity = alpha.toFixed(2);
      }
    }
  }

  dispose() {
    this.s3.scene.remove(this.group);
    for (const bd of this.bonds) { bd.mat.dispose(); bd.flash?.material.dispose(); }
    for (const p of this.pts) { p.geometry.dispose(); p.material.dispose(); }
    for (const sp of this.letters) sp.material.dispose();
    for (const c of this.charge) for (const sp of c) sp?.material.dispose();
    this.linksR.mat.dispose(); this.linksP.mat.dispose();
    this.burst.material.dispose();
    for (const l of [...this.labelsR, ...this.labelsP]) l.el.remove();
  }
}
