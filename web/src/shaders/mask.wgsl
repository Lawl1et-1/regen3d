// Alive masking (second pass of each NCA step): next -> state.
// Also applies the toxin kill, wound decay and tracks local growth (change in alpha).

struct Params {
  n: u32,
  hidden: u32,
  seed: u32,
  hiddenSel: u32,
  fireRate: f32,
  woundDecay: f32,
  toxinKill: f32,
  freeze: u32,
};

struct Ctrl {
  step: u32,
  _a: u32,
  _b: u32,
  _c: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> nxt: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(3) var<storage, read> pre: array<f32>;
@group(0) @binding(4) var<storage, read_write> wound: array<f32>;
@group(0) @binding(5) var<storage, read> toxin: array<f32>;
@group(0) @binding(6) var<storage, read_write> ctrl: Ctrl;
@group(0) @binding(7) var<storage, read_write> growth: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn mask(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.n);
  let g = vec3i(gid);
  if (any(g >= vec3i(n))) { return; }
  let vi = u32(g.x + g.y * n + g.z * n * n);
  if (all(gid == vec3u(0u))) { ctrl.step = ctrl.step + 1u; }
  var maxa = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = g + vec3i(dx, dy, dz);
        if (all(q >= vec3i(0)) && all(q < vec3i(n))) {
          maxa = max(maxa, nxt[u32(q.x + q.y * n + q.z * n * n) * 4u].x);
        }
      }
    }
  }
  let keep = pre[vi] > 0.5 && maxa > 0.1 && toxin[vi] < P.toxinKill;
  let old = state[vi * 4u].x;
  for (var k = 0u; k < 4u; k++) {
    state[vi * 4u + k] = select(vec4f(0.0), nxt[vi * 4u + k], keep);
  }
  let na = select(0.0, nxt[vi * 4u].x, keep);
  growth[vi] = mix(growth[vi], na - old, 0.25);
  wound[vi] = wound[vi] * P.woundDecay;
}
