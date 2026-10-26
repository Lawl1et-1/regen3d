// 3D Neural Cellular Automaton step, identical to train/nca.py:
//   perception = [identity(16), sobel_x(16), sobel_y(16), sobel_z(16)]
//   dx = W2 * relu(W1 * perception + b1)
//   stochastic update (fire rate), then alive masking (pre & post max-pool > 0.1).
// State layout: 16 floats per voxel as 4 x vec4, voxel index = x + y*N + z*N*N.

struct Params {
  n: u32,
  hidden: u32,
  seed: u32,
  hiddenSel: u32,
  fireRate: f32,
  woundDecay: f32,
  ischemiaRate: f32,
  freeze: u32,
  gate: u32,
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
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(3) var<storage, read> w1: array<vec4f>;
@group(0) @binding(4) var<storage, read> b1: array<f32>;
@group(0) @binding(5) var<storage, read> w2t: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> pre: array<f32>;
@group(0) @binding(7) var<storage, read_write> act: array<f32>;
@group(0) @binding(8) var<storage, read> ctrl: Ctrl;

var<workgroup> tile: array<vec4f, 864>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand01(i: u32, step: u32, seed: u32) -> f32 {
  return f32(pcg(i ^ pcg(step * 9781u + seed * 6271u)) & 0xffffffu) / 16777216.0;
}

const SM = array<f32, 3>(1.0, 2.0, 1.0);
const DF = array<f32, 3>(-1.0, 0.0, 1.0);

@compute @workgroup_size(4, 4, 4)
fn step(@builtin(workgroup_id) wg: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let n = i32(P.n);
  let base = vec3i(wg) * 4 - vec3i(1);
  for (var t = li; t < 216u; t += 64u) {
    let o = vec3i(i32(t % 6u), i32((t / 6u) % 6u), i32(t / 36u));
    let g = base + o;
    var a = vec4f(0.0); var b = vec4f(0.0); var c = vec4f(0.0); var d = vec4f(0.0);
    if (all(g >= vec3i(0)) && all(g < vec3i(n))) {
      let i = u32(g.x + g.y * n + g.z * n * n) * 4u;
      a = src[i]; b = src[i + 1u]; c = src[i + 2u]; d = src[i + 3u];
    }
    tile[t * 4u] = a; tile[t * 4u + 1u] = b; tile[t * 4u + 2u] = c; tile[t * 4u + 3u] = d;
  }
  workgroupBarrier();

  let g = vec3i(wg * 4u + lid);
  if (any(g >= vec3i(n))) { return; }
  let vi = u32(g.x + g.y * n + g.z * n * n);
  let cl = lid + vec3u(1u);

  var maxa = 0.0;
  var px = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
  var py = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
  var pz = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
  for (var dz = 0u; dz < 3u; dz++) {
    for (var dy = 0u; dy < 3u; dy++) {
      for (var dx = 0u; dx < 3u; dx++) {
        let ti = ((cl.z + dz - 1u) * 36u + (cl.y + dy - 1u) * 6u + (cl.x + dx - 1u)) * 4u;
        maxa = max(maxa, tile[ti].x);
        let wx = SM[dz] * SM[dy] * DF[dx];
        let wy = SM[dz] * DF[dy] * SM[dx];
        let wz = DF[dz] * SM[dy] * SM[dx];
        for (var k = 0u; k < 4u; k++) {
          let v = tile[ti + k];
          px[k] += v * wx; py[k] += v * wy; pz[k] += v * wz;
        }
      }
    }
  }
  let alivePre = maxa > 0.1;
  pre[vi] = select(0.0, 1.0, alivePre);
  let ci = (cl.z * 36u + cl.y * 6u + cl.x) * 4u;
  if (!alivePre || P.freeze == 1u) {
    for (var k = 0u; k < 4u; k++) { dst[vi * 4u + k] = select(vec4f(0.0), tile[ci + k], alivePre); }
    act[vi] = act[vi] * 0.9;
    return;
  }

  var p: array<vec4f, 16>;
  for (var k = 0u; k < 4u; k++) {
    p[k] = tile[ci + k];
    p[4u + k] = px[k] / 32.0;
    p[8u + k] = py[k] / 32.0;
    p[12u + k] = pz[k] / 32.0;
  }

  var upd = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
  for (var h = 0u; h < P.hidden; h++) {
    var acc = b1[h];
    let r = h * 16u;
    for (var i = 0u; i < 16u; i++) { acc += dot(w1[r + i], p[i]); }
    let hv = max(acc, 0.0);
    if (hv > 0.0) {
      let o = h * 4u;
      upd[0] += w2t[o] * hv; upd[1] += w2t[o + 1u] * hv;
      upd[2] += w2t[o + 2u] * hv; upd[3] += w2t[o + 3u] * hv;
    }
  }
  let fire = select(0.0, 1.0, rand01(vi, ctrl.step, P.seed) < P.fireRate);
  for (var k = 0u; k < 4u; k++) { dst[vi * 4u + k] = p[k] + upd[k] * fire; }
  // magnitude of the hidden-channel update: the "signals" the cells exchange
  let sig = length(upd[1].yzw) + length(upd[2]) + length(upd[3]);
  act[vi] = mix(act[vi], sig, 0.12);
}
