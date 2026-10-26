// Alive masking (second pass of each NCA step): next -> state.
// Also applies the blood-supply constraint of the simulated world (not a repair rule:
// the network still decides what to build):
//   - a dead voxel can only become tissue where blood reaches (>= 10% of its healthy supply)
//   - living tissue that loses its supply accumulates ischemia and dies after ~40 steps
// and decays the wound marker.

struct Params {
  n: u32,
  hidden: u32,
  seed: u32,
  hiddenSel: u32,
  fireRate: f32,
  woundDecay: f32,
  ischemiaRate: f32,
  freeze: u32,
  gate: u32,       // 1 = blood-supply constraint on
  _a: u32,
  _b: u32,
  _c: u32,
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
@group(0) @binding(5) var<storage, read_write> ischemia: array<f32>;
@group(0) @binding(6) var<storage, read_write> ctrl: Ctrl;
@group(0) @binding(7) var<storage, read> nut: array<f32>;
@group(0) @binding(8) var<storage, read> nutRef: array<f32>;

const SUPPLIED = 0.1;   // share of the healthy blood supply a cell needs

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
  var keep = pre[vi] > 0.5 && maxa > 0.1;
  let wasAlive = state[vi * 4u].x > 0.1;
  var isch = ischemia[vi];
  if (P.gate == 1u) {
    let r = nutRef[vi];
    let supplied = r < 1e-3 || nut[vi] >= SUPPLIED * r;   // voxels with ~no healthy supply are exempt
    if (!wasAlive && !supplied) { keep = false; }          // no new tissue without blood
    if (wasAlive && !supplied) { isch = isch + P.ischemiaRate; } else { isch = max(isch - 0.05, 0.0); }
    if (isch >= 1.0) { keep = false; }                     // starved tissue dies
  }
  if (!keep) { isch = 0.0; }
  ischemia[vi] = isch;
  for (var k = 0u; k < 4u; k++) {
    state[vi * 4u + k] = select(vec4f(0.0), nxt[vi * 4u + k], keep);
  }
  wound[vi] = wound[vi] * P.woundDecay;
}
