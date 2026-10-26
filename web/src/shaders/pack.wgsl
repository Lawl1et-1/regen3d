// Pack the simulation state into filterable textures for ray-marching, and a
// compact u32 per voxel (class | nutrient << 8 | activity << 16) for CPU metrics.

struct Pack {
  n: u32,
  hiddenSel: u32,
  _a: u32,
  _b: u32,
};

@group(0) @binding(0) var<uniform> K: Pack;
@group(0) @binding(1) var<storage, read> state: array<vec4f>;
@group(0) @binding(2) var<storage, read> wound: array<f32>;
@group(0) @binding(3) var<storage, read> nut: array<f32>;
@group(0) @binding(4) var<storage, read> act: array<f32>;
@group(0) @binding(5) var<storage, read> toxin: array<f32>;
@group(0) @binding(6) var<storage, read> growth: array<f32>;
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
  let hs = array<f32, 11>(s1.y, s1.z, s1.w, s2.x, s2.y, s2.z, s2.w, s3.x, s3.y, s3.z, s3.w);
  let h = hs[min(K.hiddenSel, 10u)];
  let a = clamp(s0.x, 0.0, 1.0);
  let types = clamp(vec4f(s0.y, s0.z, s0.w, s1.x), vec4f(0.0), vec4f(1.0));
  textureStore(tex0, gid, vec4f(a, types.x, types.y, types.z));
  textureStore(tex1, gid, vec4f(types.w, h, nut[i], act[i]));
  textureStore(tex2, gid, vec4f(wound[i], toxin[i], growth[i], 0.0));

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
