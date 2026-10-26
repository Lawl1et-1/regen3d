// HDR post: soft-knee bloom chain, subtle depth of field, ACES tone mapping,
// vignette, grain, slice flash.

struct P {
  exposure: f32,
  bloom: f32,
  focus: f32,
  aperture: f32,
  time: f32,
  flash: f32,
  vignette: f32,
  grain: f32,
};

@group(0) @binding(0) var<uniform> pu: P;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;
@group(0) @binding(4) var depthTex: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

fn down13(uv: vec2f) -> vec3f {
  let t = 1.0 / vec2f(textureDimensions(src));
  let a = textureSampleLevel(src, smp, uv + t * vec2f(-2, -2), 0.0).rgb;
  let b = textureSampleLevel(src, smp, uv + t * vec2f(0, -2), 0.0).rgb;
  let c = textureSampleLevel(src, smp, uv + t * vec2f(2, -2), 0.0).rgb;
  let d = textureSampleLevel(src, smp, uv + t * vec2f(-2, 0), 0.0).rgb;
  let e = textureSampleLevel(src, smp, uv, 0.0).rgb;
  let f = textureSampleLevel(src, smp, uv + t * vec2f(2, 0), 0.0).rgb;
  let g = textureSampleLevel(src, smp, uv + t * vec2f(-2, 2), 0.0).rgb;
  let h = textureSampleLevel(src, smp, uv + t * vec2f(0, 2), 0.0).rgb;
  let i = textureSampleLevel(src, smp, uv + t * vec2f(2, 2), 0.0).rgb;
  let j = textureSampleLevel(src, smp, uv + t * vec2f(-1, -1), 0.0).rgb;
  let k = textureSampleLevel(src, smp, uv + t * vec2f(1, -1), 0.0).rgb;
  let l = textureSampleLevel(src, smp, uv + t * vec2f(-1, 1), 0.0).rgb;
  let m = textureSampleLevel(src, smp, uv + t * vec2f(1, 1), 0.0).rgb;
  return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

@fragment
fn fsPrefilter(i: VOut) -> @location(0) vec4f {
  let c = down13(i.uv) * pu.exposure;
  let br = max(c.r, max(c.g, c.b));
  let knee = 0.5;
  let thr = 0.9;
  var soft = clamp(br - thr + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee + 1e-4);
  let contrib = max(soft, br - thr) / max(br, 1e-4);
  return vec4f(min(c * contrib, vec3f(40.0)), 1.0);
}

@fragment
fn fsDown(i: VOut) -> @location(0) vec4f { return vec4f(down13(i.uv), 1.0); }

@fragment
fn fsUp(i: VOut) -> @location(0) vec4f {
  let t = 1.0 / vec2f(textureDimensions(src));
  var c = textureSampleLevel(src, smp, i.uv, 0.0).rgb * 4.0;
  c += (textureSampleLevel(src, smp, i.uv + t * vec2f(-1, 0), 0.0).rgb + textureSampleLevel(src, smp, i.uv + t * vec2f(1, 0), 0.0).rgb
      + textureSampleLevel(src, smp, i.uv + t * vec2f(0, -1), 0.0).rgb + textureSampleLevel(src, smp, i.uv + t * vec2f(0, 1), 0.0).rgb) * 2.0;
  c += textureSampleLevel(src, smp, i.uv + t * vec2f(-1, -1), 0.0).rgb + textureSampleLevel(src, smp, i.uv + t * vec2f(1, -1), 0.0).rgb
     + textureSampleLevel(src, smp, i.uv + t * vec2f(-1, 1), 0.0).rgb + textureSampleLevel(src, smp, i.uv + t * vec2f(1, 1), 0.0).rgb;
  return vec4f(c / 16.0, 0.7);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453); }

const POISSON = array<vec2f, 12>(
  vec2f(-0.326, -0.406), vec2f(-0.840, -0.074), vec2f(-0.696, 0.457), vec2f(-0.203, 0.621),
  vec2f(0.962, -0.195), vec2f(0.473, -0.480), vec2f(0.519, 0.767), vec2f(0.185, -0.893),
  vec2f(0.507, 0.064), vec2f(0.896, 0.412), vec2f(-0.322, -0.933), vec2f(-0.792, -0.598));

@fragment
fn fsComposite(i: VOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(src));
  let d = textureSampleLevel(depthTex, smp, i.uv, 0.0).x;
  let coc = clamp(abs(d - pu.focus) / max(pu.focus, 0.1) * pu.aperture, 0.0, 1.0);
  var c = textureSampleLevel(src, smp, i.uv, 0.0).rgb;
  if (coc > 0.02) {
    let r = coc * 6.0 / dims;
    var acc = c; var wsum = 1.0;
    for (var k = 0; k < 12; k++) {
      let suv = i.uv + POISSON[k] * r;
      let sd = textureSampleLevel(depthTex, smp, suv, 0.0).x;
      let sc = clamp(abs(sd - pu.focus) / max(pu.focus, 0.1) * pu.aperture, 0.0, 1.0);
      let w = max(sc, 0.15);
      acc += textureSampleLevel(src, smp, suv, 0.0).rgb * w;
      wsum += w;
    }
    c = acc / wsum;
  }
  c = c * pu.exposure + textureSampleLevel(bloomTex, smp, i.uv, 0.0).rgb * pu.bloom;
  c += vec3f(0.8, 0.95, 1.0) * pu.flash;
  var o = aces(c);
  let v = i.uv - 0.5;
  o *= 1.0 - pu.vignette * dot(v, v) * 1.6;
  o = max(o, vec3f(0.0));
  o = pow(o, vec3f(1.0 / 2.2));
  o += (hash(i.uv * dims + pu.time * 61.0) - 0.5) * pu.grain;
  // background matches the body view (#05070b) where no tissue was hit
  let cov = textureSampleLevel(depthTex, smp, i.uv, 0.0).y;
  o += vec3f(0.0196, 0.0275, 0.0431) * (1.0 - cov);
  return vec4f(o, 1.0);
}
