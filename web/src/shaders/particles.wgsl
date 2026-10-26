// GPU particles: growth flow (streams along the direction of tissue growth),
// dissolving debris from the detached piece, and sparks along a cut.

struct Particle { pos: vec4f, vel: vec4f, col: vec4f };

struct PU {
  viewProj: mat4x4f,
  mainToWorld: mat4x4f,
  pieceToWorld: mat4x4f,
  cam: vec4f,        // eye xyz, time
  a: vec4f,          // dt, n, growthRate, debrisRate
  b: vec4f,          // spark seg start xyz, sparkCount
  c: vec4f,          // spark seg end xyz, frame
  d: vec4f,          // piece velocity xyz, render scale
  e: vec4f,          // counts: growth, debris, sparks, viewport aspect
};

@group(0) @binding(0) var<uniform> pu: PU;
@group(0) @binding(1) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(2) var mainT0: texture_3d<f32>;
@group(0) @binding(3) var mainT2: texture_3d<f32>;
@group(0) @binding(4) var pieceT0: texture_3d<f32>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(seed: ptr<function, u32>) -> f32 { *seed = pcg(*seed); return f32(*seed & 0xffffffu) / 16777216.0; }

fn loadA(t: texture_3d<f32>, g: vec3i) -> f32 {
  let n = i32(pu.a.y);
  if (any(g < vec3i(0)) || any(g >= vec3i(n))) { return 0.0; }
  return textureLoad(t, g, 0).x;
}

fn toWorld(m: mat4x4f, g: vec3f) -> vec3f {
  let lp = (g + 0.5) / pu.a.y * 2.0 - 1.0;
  return (m * vec4f(lp, 1.0)).xyz;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let nG = u32(pu.e.x); let nD = u32(pu.e.y); let nS = u32(pu.e.z);
  if (i >= nG + nD + nS) { return; }
  var p = parts[i];
  let dt = pu.a.x;
  var seed = pcg(i * 7919u + u32(pu.c.w) * 104729u);
  let n = i32(pu.a.y);
  p.pos.w -= dt;
  if (p.pos.w <= 0.0) {
    p.col = vec4f(0.0);
    if (i < nG) {
      // growth flow: try a few random voxels, spawn where alpha is increasing
      if (rnd(&seed) < pu.a.z) {
        for (var k = 0; k < 6; k++) {
          let g = vec3i(vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) * f32(n));
          let gr = textureLoad(mainT2, g, 0).z;
          if (gr > 0.004 && loadA(mainT0, g) > 0.1) {
            let grad = vec3f(loadA(mainT0, g + vec3i(1, 0, 0)) - loadA(mainT0, g - vec3i(1, 0, 0)),
                             loadA(mainT0, g + vec3i(0, 1, 0)) - loadA(mainT0, g - vec3i(0, 1, 0)),
                             loadA(mainT0, g + vec3i(0, 0, 1)) - loadA(mainT0, g - vec3i(0, 0, 1)));
            let jit = vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5;
            let wp = toWorld(pu.mainToWorld, vec3f(g) + jit);
            var dir = -grad;
            if (length(dir) < 1e-3) { dir = jit; }
            dir = normalize((pu.mainToWorld * vec4f(normalize(dir), 0.0)).xyz);
            p.pos = vec4f(wp, 0.8 + rnd(&seed) * 0.9);
            p.vel = vec4f(dir * (0.10 + 0.25 * min(gr * 20.0, 1.0)), 1.0);
            p.col = vec4f(0.45, 0.95, 1.0, 1.0) * (0.5 + min(gr * 30.0, 1.5));
            break;
          }
        }
      }
    } else if (i < nG + nD) {
      if (rnd(&seed) < pu.a.w) {
        for (var k = 0; k < 8; k++) {
          let g = vec3i(vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) * f32(n));
          let s = textureLoad(pieceT0, g, 0);
          if (s.x > 0.3) {
            let wp = toWorld(pu.pieceToWorld, vec3f(g));
            let ctr = (pu.pieceToWorld * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
            let out = normalize(wp - ctr + (vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5) * 0.6);
            p.pos = vec4f(wp, 1.2 + rnd(&seed) * 1.6);
            p.vel = vec4f(pu.d.xyz * 0.6 + out * (0.15 + rnd(&seed) * 0.35), 2.0);
            var c = vec3f(1.0, 0.55, 0.5);
            if (s.w > 0.4) { c = vec3f(1.0, 0.18, 0.2); }
            if (s.y > 0.5) { c = vec3f(1.0, 0.85, 0.8); }
            p.col = vec4f(mix(c, vec3f(0.5, 0.95, 1.0), rnd(&seed) * 0.6) * 2.2, 1.0);
            break;
          }
        }
      }
    } else {
      if (pu.b.w > 0.0 && rnd(&seed) < 0.9) {
        let t = rnd(&seed);
        let wp = mix(pu.b.xyz, pu.c.xyz, t);
        let r = normalize(vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5);
        p.pos = vec4f(wp, 0.3 + rnd(&seed) * 0.5);
        p.vel = vec4f(r * (0.4 + rnd(&seed) * 1.2), 3.0);
        p.col = vec4f(0.8, 0.97, 1.0, 1.0) * 3.0;
      }
    }
  } else {
    let kind = p.vel.w;
    var drag = 0.6;
    if (kind > 2.5) { drag = 3.5; }
    if (kind > 1.5 && kind < 2.5) { drag = 0.4; p.vel.y -= 0.05 * dt; }
    p.vel = vec4f(p.vel.xyz * exp(-drag * dt), p.vel.w);
    p.pos = vec4f(p.pos.xyz + p.vel.xyz * dt, p.pos.w);
  }
  parts[i] = p;
}

// ---- render
@group(0) @binding(0) var<uniform> ru: PU;
@group(0) @binding(1) var<storage, read> rparts: array<Particle>;
@group(0) @binding(2) var depthTex: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) q: vec2f,
  @location(1) col: vec4f,
  @location(2) dist: f32,
};

@vertex
fn vsP(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let p = rparts[ii];
  let corners = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let c = corners[vi];
  if (p.pos.w <= 0.0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  let clip = ru.viewProj * vec4f(p.pos.xyz, 1.0);
  var size = 0.006;
  if (p.vel.w > 1.5) { size = 0.011; }
  if (p.vel.w > 2.5) { size = 0.005; }
  let life = clamp(p.pos.w, 0.0, 1.0);
  o.pos = clip + vec4f(c * size * vec2f(1.0 / ru.e.w, 1.0) * 1.65, 0.0, 0.0);
  o.q = c;
  o.col = vec4f(p.col.rgb * life, 1.0);
  o.dist = length(p.pos.xyz - ru.cam.xyz);
  return o;
}

@fragment
fn fsP(i: VOut) -> @location(0) vec4f {
  let d = textureLoad(depthTex, vec2i(i.pos.xy * ru.d.w), 0).x;
  if (i.dist > d + 0.02) { discard; }
  let r = dot(i.q, i.q);
  if (r > 1.0) { discard; }
  let fall = exp(-r * 3.5);
  return vec4f(i.col.rgb * fall, 0.0);
}
