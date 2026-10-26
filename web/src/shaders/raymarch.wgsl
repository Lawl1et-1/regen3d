// Volume ray-marcher for the tissue state.
// Realistic look: trilinear density, gradient normals, approximate subsurface
// scattering (light transmittance through the tissue), wet specular on the
// capsule, ambient occlusion along the normal, emissive vessels.
// Also: cell-type map, X-ray, hidden channels, nutrient field, histology,
// forecast hologram (ghost), injury ripples and signal waves.

struct U {
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  worldToLocal: mat4x4f,
  localToWorld: mat4x4f,
  cam: vec4f,      // eye xyz, time
  res: vec4f,      // width, height, grid n, mode
  light: vec4f,    // world light dir xyz, exposure
  clip: vec4f,     // histology plane (local): normal xyz, offset
  params: vec4f,   // hiddenGain, ghostAmt, dissolve, starvedThreshold
  params2: vec4f,  // signalGain, pulseFront, pulseAmt, frame
  flags: vec4f,    // clipOn, compare, histoCuts, background
  cuts: array<vec4f, 4>,
  cutAge: vec4f,
  inj: array<vec4f, 4>,
  focus: vec4f,    // tint rgb, tintAmt (UNRECOVERABLE red hologram)
  tissue: vec4f,   // parenchyma colour of this organ
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var t0: texture_3d<f32>;
@group(0) @binding(3) var t1: texture_3d<f32>;
@group(0) @binding(4) var t2: texture_3d<f32>;
@group(0) @binding(5) var tperf: texture_3d<f32>;
@group(0) @binding(6) var g0: texture_3d<f32>;
@group(0) @binding(7) var g1: texture_3d<f32>;
@group(0) @binding(8) var depthIn: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

const CAP = vec3f(0.93, 0.80, 0.76);
const PAR = vec3f(0.86, 0.40, 0.37);
const VES = vec3f(0.42, 0.02, 0.04);
const DRN = vec3f(0.40, 0.36, 0.62);
const CT_CAP = vec3f(0.95, 0.88, 0.45);
const CT_PAR = vec3f(0.35, 0.75, 0.55);
const CT_VES = vec3f(0.92, 0.16, 0.22);
const CT_DRN = vec3f(0.30, 0.50, 0.98);
const CYAN = vec3f(0.25, 0.95, 1.0);

fn hash3(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
fn hash2(p: vec2f) -> vec2f {
  let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}
fn vnoise(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p); let w = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3f(1, 0, 0)), w.x), mix(hash3(i + vec3f(0, 1, 0)), hash3(i + vec3f(1, 1, 0)), w.x), w.y),
             mix(mix(hash3(i + vec3f(0, 0, 1)), hash3(i + vec3f(1, 0, 1)), w.x), mix(hash3(i + vec3f(0, 1, 1)), hash3(i + vec3f(1, 1, 1)), w.x), w.y), w.z);
}

fn S0(p: vec3f) -> vec4f { return textureSampleLevel(t0, smp, p * 0.5 + 0.5, 0.0); }
fn S1(p: vec3f) -> vec4f { return textureSampleLevel(t1, smp, p * 0.5 + 0.5, 0.0); }
fn S2(p: vec3f) -> vec4f { return textureSampleLevel(t2, smp, p * 0.5 + 0.5, 0.0); }
fn G0(p: vec3f) -> vec4f { return textureSampleLevel(g0, smp, p * 0.5 + 0.5, 0.0); }
fn PF(p: vec3f) -> f32 { return textureSampleLevel(tperf, smp, p * 0.5 + 0.5, 0.0).r; }

fn clipped(p: vec3f) -> bool { return u.flags.x > 0.5 && dot(p, u.clip.xyz) > u.clip.w; }
fn A(p: vec3f) -> f32 { if (clipped(p)) { return 0.0; } return S0(p).x; }

fn normalAt(p: vec3f, h: f32) -> vec3f {
  let g = vec3f(A(p + vec3f(h, 0, 0)) - A(p - vec3f(h, 0, 0)),
                A(p + vec3f(0, h, 0)) - A(p - vec3f(0, h, 0)),
                A(p + vec3f(0, 0, h)) - A(p - vec3f(0, 0, h)));
  let l = length(g);
  if (l < 1e-5) { return vec3f(0.0, 1.0, 0.0); }
  return -g / l;
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn inferno(t: f32) -> vec3f {
  let c0 = vec3f(0.0002, 0.0016, -0.0194); let c1 = vec3f(0.1065, 0.5639, 3.9327);
  let c2 = vec3f(11.6024, -3.9728, -15.9423); let c3 = vec3f(-41.7039, 17.4363, 44.3541);
  let c4 = vec3f(77.1629, -33.4023, -81.8073); let c5 = vec3f(-71.3194, 32.6260, 73.2095);
  let c6 = vec3f(25.1311, -12.2426, -23.0703);
  let x = clamp(t, 0.0, 1.0);
  return max(c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6))))), vec3f(0.0));
}

fn diverging(v: f32) -> vec3f {
  let a = clamp(v, -1.0, 1.0);
  if (a >= 0.0) { return mix(vec3f(0.08, 0.05, 0.12), vec3f(1.0, 0.62, 0.18), a); }
  return mix(vec3f(0.08, 0.05, 0.12), vec3f(0.2, 0.55, 1.0), -a);
}

fn background(uv: vec2f) -> vec3f {
  let c = uv - vec2f(0.5, 0.42);
  let r = length(c * vec2f(u.res.x / u.res.y, 1.0));
  var col = mix(vec3f(0.050, 0.055, 0.075), vec3f(0.010, 0.011, 0.016), smoothstep(0.0, 0.9, r));
  col += vec3f(0.035, 0.05, 0.075) * exp(-pow((uv.y - 0.86) * 3.0, 2.0)) * 0.5;
  return col;
}

// Voronoi for histology nuclei: returns (distance to nearest feature, cell id)
fn voronoi(q: vec2f) -> vec2f {
  let i = floor(q); let f = fract(q);
  var best = 8.0; var id = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let o = vec2f(f32(x), f32(y));
      let h = hash2(i + o);
      let r = o + 0.15 + 0.7 * h - f;
      let d = dot(r, r);
      if (d < best) { best = d; id = h.x; }
    }
  }
  return vec2f(sqrt(best), id);
}

// H&E-like stained cross-section
fn histology(p: vec3f, nrm: vec3f) -> vec3f {
  let a = S0(p); let b = S1(p);
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(nrm.y) > 0.9) { up = vec3f(1.0, 0.0, 0.0); }
  let ux = normalize(cross(up, nrm)); let uy = cross(nrm, ux);
  let q = vec2f(dot(p, ux), dot(p, uy)) * u.res.z * 0.5;
  let eosin = vec3f(0.93, 0.56, 0.70);
  let eosinDeep = vec3f(0.82, 0.36, 0.56);
  let hema = vec3f(0.30, 0.15, 0.50);
  let vor = voronoi(q * 1.7);
  let nuc = 1.0 - smoothstep(0.13, 0.2, vor.x);
  let grain = vnoise(vec3f(q * 6.0, 0.5));
  // parenchyma: pink cytoplasm with round purple nuclei, faint cell borders
  let edge = smoothstep(0.42, 0.5, vor.x);
  var par = mix(eosin * (0.92 + 0.12 * grain), eosinDeep, edge * 0.25);
  par = mix(par, hema * (0.85 + 0.3 * vor.y), nuc * 0.9);
  // capsule: dense fibrous pink with flattened nuclei
  let fib = 0.5 + 0.5 * sin(q.x * 3.0 + vnoise(vec3f(q * 1.5, 2.0)) * 6.0);
  let flat = 1.0 - smoothstep(0.1, 0.16, voronoi(q * vec2f(3.0, 1.0)).x);
  var cap = mix(eosinDeep * (0.85 + 0.2 * fib), hema, flat * 0.7);
  // vessels: lumen with red blood cells, ringed by a purple wall
  let ves = a.w;
  let lumen = smoothstep(0.62, 0.8, ves);
  let wall = smoothstep(0.22, 0.42, ves) * (1.0 - lumen);
  let rbc = 1.0 - smoothstep(0.18, 0.28, voronoi(q * 2.6 + 7.0).x);
  var vcol = mix(vec3f(0.98, 0.93, 0.95), vec3f(0.86, 0.22, 0.30), rbc * 0.85);
  // drain: clear lumen with a cuboidal purple lining
  let dr = b.x;
  let dl = smoothstep(0.6, 0.8, dr);
  let dw = smoothstep(0.25, 0.45, dr) * (1.0 - dl);
  let w = vec4f(a.y, a.z, a.w, b.x);
  var col = mix(par, cap, clamp(w.x / max(w.x + w.y, 1e-3), 0.0, 1.0) * smoothstep(0.1, 0.4, w.x));
  col = mix(col, vec3f(0.55, 0.16, 0.42), wall);
  col = mix(col, vcol, lumen);
  col = mix(col, vec3f(0.42, 0.2, 0.6), dw);
  col = mix(col, vec3f(0.97, 0.95, 0.98), dl);
  return col;
}

// Clean stylised cross-section of the tissue (no blood): soft lobules in the
// parenchyma, cut vessels as dark red rings, veins as blue-violet rings, a thin pale capsule.
fn section(p: vec3f, nrm: vec3f) -> vec3f {
  let a = S0(p); let b = S1(p);
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(nrm.y) > 0.9) { up = vec3f(1.0, 0.0, 0.0); }
  let ux = normalize(cross(up, nrm)); let uy = cross(nrm, ux);
  let q = vec2f(dot(p, ux), dot(p, uy)) * u.res.z * 0.5;
  let base = mix(u.tissue.rgb, vec3f(1.0, 0.80, 0.76), 0.32);
  let lob = voronoi(q * 0.9);
  let edge = smoothstep(0.38, 0.5, lob.x);
  let fine = vnoise(vec3f(q * 5.0, 1.0));
  var col = base * (0.94 + 0.04 * fine + 0.05 * lob.y) * (1.0 - 0.22 * edge);
  let centre = 1.0 - smoothstep(0.0, 0.12, lob.x);
  col = mix(col, u.tissue.rgb * 0.6, centre * 0.35);
  // cut vessels: a soft darker ring with a dark lumen (no blood)
  let ves = a.w;
  let wall = smoothstep(0.3, 0.5, ves) * (1.0 - smoothstep(0.6, 0.8, ves));
  let lumen = smoothstep(0.62, 0.85, ves);
  col = mix(col, u.tissue.rgb * vec3f(1.15, 0.55, 0.55), wall * 0.8);
  col = mix(col, vec3f(0.16, 0.03, 0.04), lumen * 0.9);
  let dr = b.x;
  let dwall = smoothstep(0.3, 0.5, dr) * (1.0 - smoothstep(0.6, 0.8, dr));
  col = mix(col, vec3f(0.42, 0.36, 0.55), dwall * 0.7);
  col = mix(col, vec3f(0.10, 0.08, 0.16), smoothstep(0.62, 0.85, dr) * 0.9);
  col = mix(col, min(u.tissue.rgb * 1.6 + 0.05, vec3f(1.0)), smoothstep(0.35, 0.7, a.y) * 0.3);
  return col;
}

struct March {
  col: vec3f,
  trans: f32,
  depth: f32,
};

fn march(uv: vec2f, maxDepth: f32) -> March {
  var res: March;
  res.col = vec3f(0.0); res.trans = 1.0; res.depth = 1e4;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let pn = u.invViewProj * vec4f(ndc, 0.0, 1.0);
  let pf = u.invViewProj * vec4f(ndc, 1.0, 1.0);
  let wo = u.cam.xyz;
  let wd = normalize(pf.xyz / pf.w - pn.xyz / pn.w);
  let ro = (u.worldToLocal * vec4f(wo, 1.0)).xyz;
  let rdu = (u.worldToLocal * vec4f(wd, 0.0)).xyz;
  let scale = length(rdu);
  let rd = rdu / scale;
  // ray-box
  let inv = 1.0 / rd;
  let t0b = (vec3f(-1.0) - ro) * inv; let t1b = (vec3f(1.0) - ro) * inv;
  let tmin = min(t0b, t1b); let tmax = max(t0b, t1b);
  var tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
  var tf = min(min(tmax.x, tmax.y), tmax.z);
  tf = min(tf, maxDepth * scale);
  if (tn >= tf) { return res; }

  let mode = u.res.w;
  let n = u.res.z;
  let time = u.cam.w;
  let dt = 1.0 / n;
  let Lw = normalize(u.light.xyz);
  let L = normalize((u.worldToLocal * vec4f(Lw, 0.0)).xyz);
  let V = -rd;
  let ghostAmt = u.params.y;
  let dissolve = u.params.z;

  // histology clip: start marching at the plane when the camera is on the clipped side
  var histoStart = false;
  if (u.flags.x > 0.5) {
    let dn = dot(rd, u.clip.xyz);
    let side = dot(ro, u.clip.xyz) - u.clip.w;
    let tp = -side / dn;
    if (side > 0.0) {
      if (dn >= 0.0) { return res; }
      if (tp > tn) { tn = tp; histoStart = true; }
    } else if (dn > 0.0 && tp < tf) {
      tf = tp;
    }
  }

  var t = tn + dt * hash3(vec3f(uv * u.res.xy, u.params2.w));
  var T = 1.0;
  var col = vec3f(0.0);
  var surf = false;
  var first = true;
  for (var i = 0; i < 520; i++) {
    if (t > tf || T < 0.012) { break; }
    let p = ro + rd * t;
    let s0 = S0(p);
    var dens = smoothstep(0.12, 0.5, s0.x);
    if (dissolve > 0.0) {
      dens *= smoothstep(dissolve - 0.05, dissolve + 0.05, vnoise(p * 9.0));
    }
    let gh = select(vec4f(0.0), G0(p), ghostAmt > 0.001 || u.flags.y > 0.5);

    if (first && histoStart) {
      first = false;
      if (dens > 0.25) {
        let c = histology(p, u.clip.xyz) * 1.15;
        col += T * c;
        T = 0.0;
        res.depth = t / scale;
        break;
      }
    }
    first = false;

    if (dens > 0.004) {
      let s1 = S1(p);
      let w4 = clamp(vec4f(s0.y, s0.z, s0.w, s1.x), vec4f(0.0), vec4f(1.0));
      let wn = w4 / max(dot(w4, vec4f(1.0)), 0.05);
      let pf = PF(p);
      let capCol = mix(u.tissue.rgb, vec3f(0.95, 0.85, 0.82), 0.08);
      var albedo = wn.x * capCol + wn.y * u.tissue.rgb + wn.z * VES + wn.w * DRN;
      var sigma = dot(wn, vec4f(30.0, 42.0, 55.0, 46.0)) * dens;
      var emit = vec3f(0.0);
      var specAmt = 0.0;
      var N = vec3f(0.0, 1.0, 0.0);
      let needN = T > 0.03;
      if (needN) { N = normalAt(p, 1.0 / n); }

      // fresh cut surface: clean stylised cross-section where the wound is
      var cf = 0.0;
      var cutCol = vec3f(0.0);
      if (u.flags.z > 0.5 && dens > 0.3 && !surf) {
        cf = smoothstep(0.12, 0.4, S2(p).x);
        if (cf > 0.0) {
          let lit = 0.72 + 0.4 * max(dot(N, L), 0.0);
          cutCol = section(p, N) * lit;
          emit += vec3f(0.6, 0.95, 1.0) * exp(-max(u.cutAge[0], 0.0) * 5.0) * cf * 1.2;
        }
      }

      if (mode < 0.5) {
        // ---------------- realistic
        let tl = A(p + L * 0.05) + A(p + L * 0.12) + 1.6 * A(p + L * 0.26);
        let sss = exp(-tl * vec3f(0.85, 2.3, 3.1) * 1.1);
        let wrap = max((dot(N, L) + 0.45) / 1.45, 0.0);
        let occ = clamp(1.0 - 0.55 * A(p + N * 0.07) - 0.35 * A(p + N * 0.16), 0.15, 1.0);
        let key = vec3f(1.0, 0.94, 0.86) * 1.25;
        let shadow = mix(sss, vec3f(1.0), 0.3);
        var c = albedo * (key * wrap * shadow * 0.55 + vec3f(0.30, 0.34, 0.42) * occ * 0.55);
        c += albedo * sss * key * 0.32 * (0.4 + 0.6 * wn.y + 0.8 * wn.x);
        let rim = pow(1.0 - abs(dot(N, V)), 3.0);
        c += vec3f(0.45, 0.6, 0.85) * rim * occ * 0.3 * (0.3 + wn.x);
        let perfused = smoothstep(0.0, 0.02, pf);
        let pulse = 1.0 + u.params2.z * 3.0 * exp(-pow((pf - u.params2.y) / 0.05, 2.0)) * perfused;
        emit += wn.z * vec3f(0.95, 0.04, 0.07) * 0.12 * perfused * (pulse - 1.0) * (1.0 - cf);
        emit += wn.w * vec3f(0.10, 0.12, 0.45) * 0.15 * (1.0 - cf);
        // signal waves: hidden-channel activity glowing around the wound
        let act = s1.w;
        let sig = smoothstep(0.03, 0.3, act) * u.params2.x;
        emit += vec3f(0.25, 0.85, 1.0) * sig * (0.55 + 0.45 * sin(time * 5.0 - act * 30.0)) * 0.9;
        albedo = c;
        specAmt = (0.25 + 0.75 * wn.x) * (1.0 - cf);
      } else if (mode < 1.5) {
        // ---------------- cell-type map
        let ct = wn.x * CT_CAP + wn.y * CT_PAR + wn.z * CT_VES + wn.w * CT_DRN;
        let lam = 0.35 + 0.65 * max(dot(N, L), 0.0);
        albedo = ct * lam * 1.2;
        sigma = 70.0 * dens;
      } else if (mode < 2.5) {
        // ---------------- x-ray: vascular tree only
        let edge = pow(1.0 - abs(dot(N, V)), 2.5);
        sigma = (wn.x * 0.6 + wn.y * 0.25) * dens + wn.z * 30.0 * dens + wn.w * 18.0 * dens;
        albedo = vec3f(0.0);
        let perfused = smoothstep(0.0, 0.02, pf);
        let pulse = 1.0 + u.params2.z * 3.0 * exp(-pow((pf - u.params2.y) / 0.05, 2.0)) * perfused;
        emit = wn.z * vec3f(1.6, 0.10, 0.18) * (0.35 + perfused) * pulse * 2.0
             + wn.w * vec3f(0.25, 0.45, 1.6) * 1.4
             + (wn.x + wn.y) * vec3f(0.12, 0.2, 0.32) * (0.15 + edge * 1.5) * 4.0;
      } else if (mode < 3.5) {
        // ---------------- hidden channel explorer
        let h = tanh(s1.y * u.params.x);
        sigma = 6.0 * dens + abs(h) * 22.0 * dens;
        albedo = vec3f(0.06) * (0.4 + 0.6 * max(dot(N, L), 0.0));
        emit = diverging(h) * abs(h) * 2.2;
      } else if (mode < 4.5) {
        // ---------------- nutrient field
        let nu = s1.z;
        sigma = 18.0 * dens;
        albedo = vec3f(0.04);
        emit = inferno(nu) * 1.6;
        let starved = smoothstep(0.4, 0.7, wn.y) * (1.0 - smoothstep(u.params.w * 0.7, u.params.w, nu));
        emit = mix(emit, vec3f(2.2, 0.06, 0.04) * (0.7 + 0.3 * sin(time * 6.0)), starved);
        emit += wn.z * vec3f(1.2, 0.1, 0.1) * smoothstep(0.0, 0.02, pf);
      } else {
        // ---------------- histology (outside the cut: dimmed tissue)
        let lam = 0.4 + 0.6 * max(dot(N, L), 0.0);
        albedo = mix(vec3f(0.85, 0.6, 0.72), vec3f(0.55, 0.35, 0.6), wn.z) * lam * 0.9;
      }

      if (cf > 0.0) {
        albedo = mix(albedo, cutCol, cf);
        emit = emit * (1.0 - cf);
        sigma = mix(sigma, 140.0 * dens, cf);
      }

      // injury ripples
      for (var k = 0; k < 4; k++) {
        let age = u.inj[k].w;
        if (age >= 0.0 && age < 2.0) {
          let d = length(p - u.inj[k].xyz);
          let r = age * 0.55;
          let ring = exp(-pow((d - r) / 0.035, 2.0)) * (1.0 - age / 2.0);
          let flash = exp(-d * 18.0) * exp(-age * 9.0);
          emit += vec3f(0.55, 0.95, 1.0) * (ring * 2.5 + flash * 14.0);
        }
      }

      // compare: highlight differences with the forecast
      if (u.flags.y > 0.5) {
        let diff = abs(smoothstep(0.12, 0.5, gh.x) - dens);
        emit += vec3f(1.0, 0.2, 0.9) * smoothstep(0.25, 0.6, diff) * 2.5;
      }

      let alpha = 1.0 - exp(-sigma * dt);
      if (alpha > 0.0001) {
        col += T * alpha * (albedo + emit);
        // wet specular layer at the first surface crossing
        if (!surf && dens > 0.45 && T > 0.02) {
          surf = true;
          res.depth = t / scale;
          if (mode < 0.5 || mode > 4.5) {
            let wet = normalize(N + (vec3f(vnoise(p * 40.0), vnoise(p * 40.0 + 7.0), vnoise(p * 40.0 + 13.0)) - 0.5) * 0.08);
            let H = normalize(L + V);
            let ndh = max(dot(wet, H), 0.0);
            let ndv = max(dot(N, V), 0.0);
            let fres = 0.03 + 0.97 * pow(1.0 - ndv, 5.0);
            let r2 = 0.03;
            let dd = ndh * ndh * (r2 - 1.0) + 1.0;
            let ggx = r2 / (3.14159 * dd * dd);
            let r2b = 0.25;
            let ddb = ndh * ndh * (r2b - 1.0) + 1.0;
            let broad = r2b / (3.14159 * ddb * ddb);
            let fill = pow(max(dot(reflect(-V, N), normalize(vec3f(-0.4, 0.8, 0.5))), 0.0), 12.0);
            col += T * (vec3f(1.0, 0.97, 0.93) * (ggx * 0.25 + broad * 0.35) * (0.2 + fres) * specAmt * max(dot(N, L), 0.0) * 2.0
                       + vec3f(0.5, 0.6, 0.75) * fill * fres * specAmt * 1.5);
          }
        }
        T *= 1.0 - alpha;
      }
    }

    // forecast hologram: shell + scanlines where tissue is still missing
    if (ghostAmt > 0.001) {
      let missing = 1.0 - smoothstep(0.15, 0.45, s0.x);
      let shell = exp(-pow((gh.x - 0.42) / 0.11, 2.0)) * missing;
      let wp = (u.localToWorld * vec4f(p, 1.0)).xyz;
      let scan = 0.45 + 0.55 * smoothstep(0.35, 0.6, fract(wp.y * 42.0 - time * 1.4));
      let shimmer = 0.75 + 0.25 * vnoise(vec3f(p * 14.0 + time * 2.0));
      let tint = mix(CYAN, u.focus.xyz, u.focus.w);
      var gem = tint * (shell * 1.5 + smoothstep(0.3, 0.6, gh.x) * missing * 0.08) * scan * shimmer;
      // vessel plan: forecast vessels not yet present, drawn as travelling dots
      let vp = smoothstep(0.3, 0.55, gh.w) * (1.0 - smoothstep(0.15, 0.45, s0.w)) * smoothstep(0.2, 0.5, gh.x);
      let dots = smoothstep(0.62, 0.85, 0.5 + 0.5 * sin(dot(p, vec3f(1.0, 0.8, 0.6)) * n * 0.75 - time * 7.0));
      gem += mix(vec3f(0.7, 1.0, 1.0), tint, 0.3) * vp * dots * 3.0;
      col += T * gem * ghostAmt * dt * 22.0;
    }

    var stepMul = 1.0;
    if (s0.x < 0.03 && gh.x < 0.03) { stepMul = 2.0; }
    t += dt * stepMul;
  }
  res.col = col;
  res.trans = T;
  return res;
}

struct FOut { @location(0) color: vec4f, @location(1) depth: vec4f };

@fragment
fn fsMain(i: VOut) -> FOut {
  let m = march(i.uv, 1e4);
  var o: FOut;
  var bg = vec3f(0.0);
  if (u.flags.w > 0.5) { bg = background(i.uv); }
  o.color = vec4f(m.col + m.trans * bg, 1.0);
  o.depth = vec4f(m.depth, 1.0 - m.trans, 0.0, 1.0);
  return o;
}

// second volume (detached piece / planarian twin): premultiplied, stops at the main depth
@fragment
fn fsPiece(i: VOut) -> @location(0) vec4f {
  let dim = textureDimensions(depthIn);
  let d = textureLoad(depthIn, vec2i(i.uv * vec2f(dim)), 0).x;
  let m = march(i.uv, d);
  return vec4f(m.col, 1.0 - m.trans);
}
