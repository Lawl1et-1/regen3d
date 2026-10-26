// Pack the simulation state into filterable textures for ray-marching, and a
// compact u32 per voxel (class | nutrient << 8 | activity << 16) for CPU metrics.

struct Pack {
  n: u32,
  brain0: u32,   // the three hidden channels shown by the AI Brain view (state indices 5..15)
  brain1: u32,
  brain2: u32,
};

@group(0) @binding(0) var<uniform> K: Pack;
@group(0) @binding(1) var<storage, read> state: array<vec4f>;
@group(0) @binding(2) var<storage, read> wound: array<f32>;
@group(0) @binding(3) var<storage, read> nut: array<f32>;
@group(0) @binding(4) var<storage, read> act: array<f32>;
@group(0) @binding(5) var<storage, read> nutRef: array<f32>;
@group(0) @binding(7) var tex0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var tex1: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var tex2: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var<storage, read_write> summary: array<u32>;

@compute @workgroup_size(4, 4, 4)
fn pack(@builtin(global_invocation_id) gid: vec3u) {
  let n = K.n;
  if (any(gid >= vec3u(n))) { return; }
  let i = gid.x + gid.y * n + gid.z * n * n;
  let s0 = state[i * 4u];
  let s1 = state[i * 4u + 1u];
  let s2 = state[i * 4u + 2u];
  let s3 = state[i * 4u + 3u];
  let ch = array<f32, 16>(s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w, s2.x, s2.y, s2.z, s2.w, s3.x, s3.y, s3.z, s3.w);
  let h = ch[min(K.brain0, 15u)];
  let brain = tanh(vec3f(ch[min(K.brain0, 15u)], ch[min(K.brain1, 15u)], ch[min(K.brain2, 15u)]));
  let a = clamp(s0.x, 0.0, 1.0);
  let types = clamp(vec4f(s0.y, s0.z, s0.w, s1.x), vec4f(0.0), vec4f(1.0));
  // render-only smoothing: a 3x3x3 binomial filter so the 48^3 grid shows no voxel steps
  var sa = vec4f(0.0);
  var sd = 0.0;
  var swd = 0.0;
  let w3 = array<f32, 3>(1.0, 2.0, 1.0);
  for (var dz = 0; dz < 3; dz++) {
    for (var dy = 0; dy < 3; dy++) {
      for (var dx = 0; dx < 3; dx++) {
        let q = vec3i(gid) + vec3i(dx - 1, dy - 1, dz - 1);
        if (any(q < vec3i(0)) || any(q >= vec3i(i32(n)))) { continue; }
        let j = u32(q.x) + u32(q.y) * n + u32(q.z) * n * n;
        let w = w3[dx] * w3[dy] * w3[dz];
        let v0 = state[j * 4u];
        sa += clamp(v0, vec4f(0.0), vec4f(1.0)) * w;
        sd += clamp(state[j * 4u + 1u].x, 0.0, 1.0) * w;
        swd += wound[j] * w;
      }
    }
  }
  sa = sa / 64.0;
  sd = sd / 64.0;
  // density gets a wider 5x5x5 binomial filter: oblique cut walls stay smooth on a 48^3 grid
  var a5 = 0.0;
  let w5 = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
  for (var dz = 0; dz < 5; dz++) {
    for (var dy = 0; dy < 5; dy++) {
      for (var dx = 0; dx < 5; dx++) {
        let q = vec3i(gid) + vec3i(dx - 2, dy - 2, dz - 2);
        if (any(q < vec3i(0)) || any(q >= vec3i(i32(n)))) { continue; }
        let j = u32(q.x) + u32(q.y) * n + u32(q.z) * n * n;
        a5 += clamp(state[j * 4u].x, 0.0, 1.0) * w5[dx] * w5[dy] * w5[dz];
      }
    }
  }
  sa.x = a5 / 4096.0;
  // thin structures (1-voxel vessels) would fade under the filter: keep part of their own value
  let va = max(sa.x, a * 0.6);
  textureStore(tex0, gid, vec4f(va, sa.y, sa.z, sa.w));
  // blood supply as a share of the healthy supply (1 = normal; voxels with no healthy supply count as normal)
  let healthyNut = nutRef[i];
  let supply = select(clamp(nut[i] / healthyNut, 0.0, 1.0), 1.0, healthyNut < 1e-3);
  textureStore(tex1, gid, vec4f(sd, h, supply, act[i]));
  textureStore(tex2, gid, vec4f(max(wound[i], swd / 64.0 * 1.6), brain));

  var cls = 0u;
  if (s0.x > 0.1) {
    var best = types.x; cls = 1u;
    if (types.y > best) { best = types.y; cls = 2u; }
    if (types.z > best) { best = types.z; cls = 3u; }
    if (types.w > best) { best = types.w; cls = 4u; }
  }
  let nq = u32(clamp(nut[i], 0.0, 1.0) * 255.0);
  let aq = u32(clamp(act[i] * 4.0, 0.0, 1.0) * 255.0);
  summary[i] = cls | (nq << 8u) | (aq << 16u);
}
