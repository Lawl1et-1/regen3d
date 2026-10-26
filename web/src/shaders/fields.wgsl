// Per-voxel field kernels: damage tools, nutrient diffusion, toxin spread,
// packing the state into filterable 3D textures for the renderer, and a compact
// summary (cell class + nutrient) read back on the CPU for metrics.

struct Field {
  n: u32,
  kind: u32,       // tool kind (see TOOL_* in sim.ts)
  hiddenSel: u32,
  flags: u32,
  p0: vec4f,       // centre xyz (voxels), radius
  p1: vec4f,       // segment end xyz / plane normal xyz + offset
  p2: vec4f,       // nutrient: lambda, toxin: diffusion, decay, amount
  p3: vec4f,       // wedge: depth direction xyz (into tissue), s0
  p4: vec4f,       // wedge: stroke direction xyz, s1
};

@group(0) @binding(0) var<uniform> F: Field;
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> wound: array<f32>;
@group(0) @binding(3) var<storage, read_write> toxinA: array<f32>;
@group(0) @binding(4) var<storage, read_write> toxinB: array<f32>;
@group(0) @binding(5) var<storage, read_write> nutA: array<f32>;
@group(0) @binding(6) var<storage, read_write> nutB: array<f32>;
@group(0) @binding(7) var<storage, read_write> act: array<f32>;
@group(0) @binding(8) var perf: texture_3d<f32>;

fn idx(g: vec3i) -> u32 { let n = i32(F.n); return u32(g.x + g.y * n + g.z * n * n); }
fn inside(g: vec3i) -> bool { return all(g >= vec3i(0)) && all(g < vec3i(i32(F.n))); }
fn alphaAt(g: vec3i) -> f32 { if (!inside(g)) { return 0.0; } return state[idx(g) * 4u].x; }

fn segDist(p: vec3f, a: vec3f, b: vec3f) -> f32 {
  let ab = b - a;
  let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
  return length(p - (a + ab * t));
}

// Swipe incision (same definition as train/damage.py::wedge): a slab around the
// blade plane, entering from the surface point e along v to a given depth,
// limited to [s0, s1] along the stroke, narrowing towards the bottom of the cut.
fn wedgeDist(p: vec3f) -> f32 {
  let d = p - F.p0.xyz;
  let depth = F.p0.w;
  let t = dot(d, F.p3.xyz);
  let s = dot(d, F.p4.xyz);
  let a = abs(dot(d, F.p1.xyz));
  let w = F.p1.w * (1.0 - 0.7 * clamp(t / depth, 0.0, 1.0));
  return max(max(a - w, t - depth), max(F.p3.w - s, s - F.p4.w));
}

// signed distance to the removed region (negative = removed)
fn toolDist(p: vec3f) -> f32 {
  switch (F.kind) {
    case 1u: { return length(p - F.p0.xyz) - F.p0.w; }                 // sphere
    case 2u: { return segDist(p, F.p0.xyz, F.p1.xyz) - F.p0.w; }       // scalpel stroke (capsule)
    case 3u: { return F.p1.w - dot(p, F.p1.xyz); }                     // half-space: removes dot(p,n) > d
    case 5u: { return wedgeDist(p); }                                  // swipe incision
    default: { return 1e9; }
  }
}

@compute @workgroup_size(4, 4, 4)
fn tool(@builtin(global_invocation_id) gid: vec3u) {
  let g = vec3i(gid);
  if (!inside(g)) { return; }
  let i = idx(g);
  let p = vec3f(g);
  if (F.kind == 4u) {  // toxin injection
    let d = length(p - F.p0.xyz);
    let amt = F.p2.z * exp(-d * d / (F.p0.w * F.p0.w));
    if (state[i * 4u].x > 0.1) { toxinA[i] = toxinA[i] + amt; }
    return;
  }
  let sd = toolDist(p);
  if (sd < 0.0) {
    for (var k = 0u; k < 4u; k++) { state[i * 4u + k] = vec4f(0.0); }
    wound[i] = 0.0; act[i] = 0.0; toxinA[i] = 0.0;
  } else if (sd < 1.6 && state[i * 4u].x > 0.1) {
    wound[i] = max(wound[i], 1.0 - sd / 1.6);
  }
}

// one Jacobi iteration of steady-state diffusion with consumption:
//   perfused vessel voxels are sources (n = 1); tissue: n = sum(tissue neighbours) / (6 + lambda);
//   an empty voxel gets what leaks from the tissue around it (so growth can only advance
//   from walls that are themselves supplied)
@compute @workgroup_size(4, 4, 4)
fn nutrient(@builtin(global_invocation_id) gid: vec3u) {
  let g = vec3i(gid);
  if (!inside(g)) { return; }
  let i = idx(g);
  let a = state[i * 4u].x;
  var v = 0.0;
  if (textureLoad(perf, g, 0).r > 0.0) {
    v = 1.0;
  } else if (a > 0.1) {
    var s = 0.0;
    let o = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
    for (var k = 0u; k < 6u; k++) {
      let q = g + o[k];
      if (inside(q) && alphaAt(q) > 0.1) { s += nutA[idx(q)]; }
    }
    v = s / (6.0 + F.p2.x);
  } else {
    var s = 0.0;
    let o = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
    for (var k = 0u; k < 6u; k++) {
      let q = g + o[k];
      if (inside(q) && alphaAt(q) > 0.1) { s += nutA[idx(q)]; }
    }
    v = s / (6.0 + F.p2.x);
  }
  nutB[i] = v;
}

// toxin spreads through living tissue and slowly decays; the mask pass kills cells above threshold
@compute @workgroup_size(4, 4, 4)
fn toxin(@builtin(global_invocation_id) gid: vec3u) {
  let g = vec3i(gid);
  if (!inside(g)) { return; }
  let i = idx(g);
  let t = toxinA[i];
  var lap = 0.0;
  let o = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
  for (var k = 0u; k < 6u; k++) {
    let q = g + o[k];
    if (inside(q)) { lap += toxinA[idx(q)] - t; }
  }
  var nt = (t + F.p2.y * lap) * F.p2.w;
  if (nt > 0.25 && state[i * 4u].x > 0.1) { wound[i] = max(wound[i], 0.8); }
  toxinB[i] = max(nt, 0.0);
}
