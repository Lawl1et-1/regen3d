// Step 1: the 3D body (BodyParts3D meshes), X-ray mode, organ hover/selection
// and a smooth camera zoom that isolates the chosen organ at the centre.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

export type OrganId = 'liver' | 'heart' | 'kidney' | 'lung' | 'stomach' | 'spleen' | 'pancreas' | 'brain' | 'bone' | 'skin';

export const ORGANS: Record<OrganId, { label: string; nodes: string[]; color: [number, number, number]; glow: [number, number, number] }> = {
  liver: { label: 'Liver', nodes: ['liver'], color: [0.40, 0.10, 0.08], glow: [1.0, 0.42, 0.25] },
  heart: { label: 'Heart', nodes: ['heart'], color: [0.62, 0.10, 0.12], glow: [1.0, 0.18, 0.25] },
  kidney: { label: 'Kidneys', nodes: ['kidney_left', 'kidney_right'], color: [0.48, 0.14, 0.14], glow: [1.0, 0.55, 0.3] },
  lung: { label: 'Lungs', nodes: ['lung_left', 'lung_right'], color: [0.80, 0.50, 0.52], glow: [0.55, 0.75, 1.0] },
  stomach: { label: 'Stomach', nodes: ['stomach'], color: [0.66, 0.36, 0.34], glow: [1.0, 0.6, 0.45] },
  spleen: { label: 'Spleen', nodes: ['spleen'], color: [0.42, 0.10, 0.18], glow: [0.9, 0.3, 0.55] },
  pancreas: { label: 'Pancreas', nodes: ['pancreas'], color: [0.78, 0.60, 0.46], glow: [1.0, 0.8, 0.45] },
  brain: { label: 'Brain', nodes: ['brain'], color: [0.78, 0.58, 0.58], glow: [0.75, 0.6, 1.0] },
  bone: { label: 'Femur (hip end)', nodes: ['femur_right', 'femur_left'], color: [0.86, 0.82, 0.72], glow: [1.0, 0.95, 0.8] },
  // the skin mesh keeps its own material: it is picked where no organ is hit, and a small patch is repaired
  skin: { label: 'Skin', nodes: [], color: [0.74, 0.58, 0.52], glow: [1.0, 0.8, 0.7] },
};

/** Size of the skin patch the camera zooms onto (m). */
const SKIN_PATCH = 0.03;

const skinVert = /* glsl */ `
  varying vec3 vN; varying vec3 vV; varying vec3 vW;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vW = wp.xyz;
    vN = normalize(mat3(modelMatrix) * normal);
    vV = normalize(cameraPosition - wp.xyz);
    gl_Position = projectionMatrix * viewMatrix * wp;
  }`;

// Skin: soft lit translucent tissue that turns into a glassy fresnel ghost in X-ray.
const skinFrag = /* glsl */ `
  uniform float uX; uniform float uFade;
  varying vec3 vN; varying vec3 vV; varying vec3 vW;
  void main() {
    vec3 n = normalize(vN); if (!gl_FrontFacing) n = -n;
    float ndv = clamp(dot(n, vV), 0.0, 1.0);
    float fres = pow(1.0 - ndv, 2.4);
    vec3 L = normalize(vec3(-0.4, 0.8, 0.6));
    float dif = max(dot(n, L), 0.0) * 0.75 + 0.25;
    vec3 skin = vec3(0.74, 0.58, 0.52) * dif + vec3(0.55, 0.65, 0.8) * fres * 0.4;
    float aSkin = mix(0.16, 0.85, fres);
    vec3 ghost = vec3(0.35, 0.75, 1.0) * (fres * 1.1 + 0.02);
    float scan = 0.85 + 0.15 * smoothstep(0.4, 0.6, fract(vW.y * 18.0));
    float aGhost = (fres * 0.32 + 0.015) * scan;
    vec3 c = mix(skin, ghost, uX);
    float a = mix(aSkin, aGhost, uX) * uFade;
    gl_FragColor = vec4(c * mix(1.0, scan, uX), a);
  }`;

// Organs: glossy wet tissue; in X-ray they glow from within (fresnel + emissive).
const organFrag = /* glsl */ `
  uniform vec3 uColor; uniform vec3 uGlow; uniform float uX; uniform float uHover; uniform float uFade;
  varying vec3 vN; varying vec3 vV; varying vec3 vW;
  void main() {
    vec3 n = normalize(vN); if (!gl_FrontFacing) n = -n;
    float ndv = clamp(dot(n, vV), 0.0, 1.0);
    float fres = pow(1.0 - ndv, 3.0);
    vec3 L = normalize(vec3(-0.4, 0.8, 0.6));
    vec3 L2 = normalize(vec3(0.6, 0.2, -0.7));
    float wrap = max((dot(n, L) + 0.4) / 1.4, 0.0);
    vec3 H = normalize(L + vV);
    float spec = pow(max(dot(n, H), 0.0), 70.0) * 0.9 + pow(max(dot(n, H), 0.0), 12.0) * 0.08;
    vec3 base = uColor * (0.22 + 0.95 * wrap) + uColor * vec3(1.0, 0.5, 0.45) * 0.25 * (1.0 - ndv);
    base += vec3(1.0, 0.96, 0.92) * spec * (0.2 + fres);
    base += vec3(0.5, 0.65, 0.9) * max(dot(n, L2), 0.0) * fres * 0.6;
    vec3 glow = uGlow * (0.08 + 1.1 * fres);
    vec3 c = base * (1.0 - 0.35 * uX) + glow * (uX * 1.05 + uHover * 0.6);
    gl_FragColor = vec4(c, uFade);
  }`;

const muscleFrag = /* glsl */ `
  uniform float uFade;
  varying vec3 vN; varying vec3 vV; varying vec3 vW;
  void main() {
    vec3 n = normalize(vN); if (!gl_FrontFacing) n = -n;
    float ndv = clamp(dot(n, vV), 0.0, 1.0);
    vec3 L = normalize(vec3(-0.4, 0.8, 0.6));
    float dif = max(dot(n, L), 0.0) * 0.8 + 0.2;
    float stri = 0.9 + 0.1 * sin(vW.y * 900.0 + vW.x * 300.0);
    vec3 c = vec3(0.50, 0.13, 0.12) * dif * stri + vec3(0.9, 0.4, 0.35) * pow(1.0 - ndv, 3.0) * 0.25;
    gl_FragColor = vec4(c, uFade);
  }`;

/** One organ mesh (a kidney or a lung is one of a pair): its own material so it can be isolated. */
interface OrganEntry { id: OrganId; node: string; mesh: THREE.Mesh; mat: THREE.ShaderMaterial; center: THREE.Vector3; radius: number; box: THREE.Box3 }

export class BodyView3D {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(32, 1, 0.01, 50);
  readonly controls: OrbitControls;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private skinMat: THREE.ShaderMaterial;
  private muscleMat: THREE.ShaderMaterial;
  private organs: OrganEntry[] = [];
  private raycaster = new THREE.Raycaster();
  private clock = new THREE.Clock();
  private xray = 0;
  xrayTarget = 0;
  hovered: OrganId | null = null;
  /** where the skin was last picked (world point and outward normal) */
  skinHit: { point: [number, number, number]; normal: [number, number, number] } | null = null;
  private skinMesh: THREE.Mesh | null = null;
  focused: OrganId | null = null;
  private focusAmt = 0; // 0 = whole body, 1 = organ isolated
  private tween: { from: [THREE.Vector3, THREE.Vector3]; to: [THREE.Vector3, THREE.Vector3]; start: number; dur: number } | null = null;
  private home = { pos: new THREE.Vector3(0, 0.3, 3.9), target: new THREE.Vector3(0, 0.12, 0) };
  onFrame?: (dt: number) => void;
  /** The scene is only re-rendered when something changed (camera, fades, hover): idle costs nothing. */
  dirty = true;
  /** When false the mesh scene is not drawn (the GPU organ volume covers it). */
  renderEnabled = true;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.25));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.scene.background = new THREE.Color(0x05070b);
    this.camera.position.copy(this.home.pos);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.target.copy(this.home.target);
    this.controls.minDistance = 0.35;
    this.controls.maxDistance = 6;
    this.controls.addEventListener('change', () => { this.dirty = true; });

    const u = () => ({ uX: { value: 0 }, uFade: { value: 1 } });
    this.skinMat = new THREE.ShaderMaterial({
      uniforms: u(), vertexShader: skinVert, fragmentShader: skinFrag,
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
    });
    this.muscleMat = new THREE.ShaderMaterial({
      uniforms: u(), vertexShader: skinVert, fragmentShader: muscleFrag, transparent: true, side: THREE.DoubleSide,
    });

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.4, 0.5, 0.7);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.addFloorGlow();
    this.resize();
    addEventListener('resize', () => this.resize());
  }

  private addFloorGlow() {
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uX: { value: 0 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: `uniform float uX; varying vec2 vUv;
        void main(){ float r = length(vUv - 0.5) * 2.0; float a = exp(-r*r*5.0) * 0.16 + exp(-pow((r-0.55)/0.012,2.0)) * 0.05 * uX;
          gl_FragColor = vec4(mix(vec3(0.35,0.45,0.6), vec3(0.3,0.8,1.0), uX) * a, 1.0); }`,
    });
    const disk = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 2.2), mat);
    disk.rotation.x = -Math.PI / 2;
    disk.position.y = -0.92;
    disk.name = 'floor';
    this.scene.add(disk);
  }

  async load(url: string) {
    const gltf = await new GLTFLoader().loadAsync(url);
    const root = gltf.scene;
    const byName = (n: string) => root.getObjectByName(n) as THREE.Mesh | undefined;
    for (const name of ['skin']) { const m = byName(name); if (m) { m.material = this.skinMat; m.renderOrder = 10; this.skinMesh = m; } }
    const muscles = byName('muscles');
    if (muscles) { muscles.material = this.muscleMat; muscles.renderOrder = 5; }
    for (const [id, o] of Object.entries(ORGANS) as [OrganId, (typeof ORGANS)[OrganId]][]) {
      for (const node of o.nodes) {
        const mesh = byName(node);
        if (!mesh) continue;
        const mat = new THREE.ShaderMaterial({
          uniforms: {
            uColor: { value: new THREE.Vector3(...o.color) }, uGlow: { value: new THREE.Vector3(...o.glow) },
            uX: { value: 0 }, uHover: { value: 0 }, uFade: { value: 1 },
          },
          vertexShader: skinVert, fragmentShader: organFrag, transparent: true, side: THREE.DoubleSide,
        });
        mesh.material = mat;
        mesh.renderOrder = 1;
        mesh.userData.organ = id;
        mesh.userData.node = node;
        const box = new THREE.Box3().expandByObject(mesh);
        const sphere = box.getBoundingSphere(new THREE.Sphere());
        this.organs.push({ id, node, mesh, mat, center: sphere.center, radius: sphere.radius, box });
      }
    }
    this.scene.add(root);
  }

  resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.setSize(Math.ceil(w / 2), Math.ceil(h / 2));
    this.dirty = true;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** The mesh entries of an organ (two for kidneys and lungs). */
  entries(id: OrganId) { return this.organs.filter((o) => o.id === id); }
  /** Mesh node under the cursor at the last pick (e.g. 'kidney_right'). */
  pickedNode: string | null = null;
  focusedNode: string | null = null;
  /** Centre of a mesh's bounding box (where a GPU volume of it is placed). */
  nodeCenter(node: string): [number, number, number] {
    const c = this.organs.find((o) => o.node === node)!.box.getCenter(new THREE.Vector3());
    return [c.x, c.y, c.z];
  }

  /** Organ under a screen point (organs are pickable even behind the skin). */
  pick(x: number, y: number): OrganId | null {
    if (this.focused) return null;
    const ndc = new THREE.Vector2((x / innerWidth) * 2 - 1, -(y / innerHeight) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObjects(this.organs.map((o) => o.mesh), false);
    this.pickedNode = (hits[0]?.object.userData.node as string) ?? null;
    if (hits[0]) return hits[0].object.userData.organ as OrganId;
    // no organ under the cursor: the skin itself (only while it is opaque)
    if (!this.skinMesh || this.xrayTarget > 0.5) return null;
    const s = this.raycaster.intersectObject(this.skinMesh, false)[0];
    if (!s?.face) return null;
    const nrm = s.face.normal.clone().transformDirection(this.skinMesh.matrixWorld);
    if (nrm.dot(this.raycaster.ray.direction) > 0) nrm.negate();
    this.skinHit = { point: [s.point.x, s.point.y, s.point.z], normal: [nrm.x, nrm.y, nrm.z] };
    this.pickedNode = 'skin';
    return 'skin';
  }

  /** Centre and radius the camera frames for an organ mesh; `topCrop` keeps only its upper part (m). */
  private frameOf(o: OrganEntry, topCrop?: number) {
    if (!topCrop) return { center: o.center.clone(), radius: o.radius };
    const b = o.box.clone();
    b.min.y = Math.max(b.min.y, b.max.y - topCrop);
    const s = b.getBoundingSphere(new THREE.Sphere());
    return { center: s.center, radius: s.radius };
  }

  /** Screen position of an organ's centre (for labels). */
  screenOf(id: OrganId): [number, number] {
    if (id === 'skin') {
      const p = new THREE.Vector3(...(this.skinHit?.point ?? [0, 0, 0])).project(this.camera);
      return [(p.x * 0.5 + 0.5) * innerWidth, (0.5 - p.y * 0.5) * innerHeight];
    }
    const es = this.entries(id);
    const p = es.reduce((a, e) => a.add(e.center), new THREE.Vector3()).multiplyScalar(1 / es.length).project(this.camera);
    return [(p.x * 0.5 + 0.5) * innerWidth, (0.5 - p.y * 0.5) * innerHeight];
  }

  /** Zoom onto one organ mesh (`node`, e.g. the kidney that was clicked) and fade everything else. */
  focus(id: OrganId, node?: string, topCrop?: number) {
    let center: THREE.Vector3, radius: number;
    let dir = this.camera.position.clone().sub(this.controls.target).setY(0).normalize();
    if (dir.lengthSq() < 0.5) dir.set(0, 0, 1);
    dir.lerp(new THREE.Vector3(0, 0, 1), 0.6).normalize().add(new THREE.Vector3(0, 0.12, 0)).normalize();
    if (id === 'skin' && this.skinHit) {
      // look straight down onto the clicked patch of skin
      center = new THREE.Vector3(...this.skinHit.point);
      radius = SKIN_PATCH;
      dir = new THREE.Vector3(...this.skinHit.normal);
      this.focusedNode = 'skin';
    } else {
      const es = this.entries(id);
      const o = es.find((e) => e.node === node) ?? es[0];
      this.focusedNode = o.node;
      ({ center, radius } = this.frameOf(o, topCrop));
    }
    this.focused = id;
    const dist = (radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov) / 2)) * 1.25;
    this.startTween(center.clone().add(dir.multiplyScalar(dist)), center.clone(), 1.6);
    this.controls.minDistance = dist * 0.45;
    this.controls.maxDistance = dist * 2.2;
  }

  unfocus() {
    this.focused = null;
    this.controls.minDistance = 0.35;
    this.controls.maxDistance = 6;
    this.startTween(this.home.pos.clone(), this.home.target.clone(), 1.4);
  }

  private startTween(pos: THREE.Vector3, target: THREE.Vector3, dur: number) {
    this.tween = { from: [this.camera.position.clone(), this.controls.target.clone()], to: [pos, target], start: performance.now(), dur };
  }

  get tweening() { return !!this.tween; }

  /** Move the camera closer (f < 1) or further (f > 1) from its target, smoothly. */
  zoomBy(f: number) {
    const t = this.controls.target.clone();
    const pos = this.camera.position.clone().sub(t).multiplyScalar(f).add(t);
    this.controls.maxDistance = Math.max(this.controls.maxDistance, pos.distanceTo(t) * 1.3);
    this.startTween(pos, t, 0.9);
  }

  set renderOn(on: boolean) { this.renderEnabled = on; this.dirty = true; }

  /** Left drag cuts (handled outside), right drag rotates; one finger cuts, two fingers rotate/zoom. */
  setCutControls(on: boolean) {
    this.controls.mouseButtons = on
      ? { LEFT: null as unknown as THREE.MOUSE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE }
      : { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this.controls.touches = on
      ? { ONE: null as unknown as THREE.TOUCH, TWO: THREE.TOUCH.DOLLY_ROTATE }
      : { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  }

  frame() {
    const dt = Math.min(this.clock.getDelta(), 0.5);
    const settle = (cur: number, goal: number, rate: number) => {
      if (Math.abs(goal - cur) < 0.002) return goal;
      this.dirty = true;
      return cur + (goal - cur) * (1 - Math.exp(-dt * rate));
    };
    this.xray = settle(this.xray, this.xrayTarget, 5);
    this.focusAmt = settle(this.focusAmt, this.focused ? 1 : 0, 3.2);

    if (this.tween) {
      const tw = this.tween;
      const t = Math.min(1, (performance.now() - tw.start) / 1000 / tw.dur);
      const e = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
      this.camera.position.lerpVectors(tw.from[0], tw.to[0], e);
      this.controls.target.lerpVectors(tw.from[1], tw.to[1], e);
      if (t >= 1) this.tween = null;
      this.dirty = true;
    }
    this.controls.enabled = !this.tween;
    this.controls.update();

    const bodyFade = 1 - this.focusAmt;
    // X-ray (or isolating an organ) dissolves the skin into a ghost and removes the muscles
    // zooming onto the skin keeps it solid
    const onSkin = this.focused === 'skin';
    const xr = onSkin ? this.xray : Math.max(this.xray, this.focusAmt);
    this.skinMat.uniforms.uX.value = xr;
    this.skinMat.uniforms.uFade.value = onSkin ? 1 : bodyFade;
    this.muscleMat.uniforms.uFade.value = (1 - this.xray) * bodyFade;
    this.muscleMat.visible = this.muscleMat.uniforms.uFade.value > 0.01;
    this.muscleMat.depthWrite = this.muscleMat.uniforms.uFade.value > 0.95;
    this.skinMat.visible = this.skinMat.uniforms.uFade.value > 0.01;
    for (const o of this.organs) {
      const isFocus = this.focused === o.id && o.node === this.focusedNode;
      const u = o.mat.uniforms;
      u.uX.value = this.xray * (1 - this.focusAmt);
      const hv = this.hovered === o.id ? 1 : 0;
      u.uHover.value = settle(u.uHover.value, hv, 10);
      u.uFade.value = isFocus ? 1 : bodyFade;
      o.mat.visible = u.uFade.value > 0.01;
      o.mat.depthWrite = u.uFade.value > 0.95;
    }
    const floor = this.scene.getObjectByName('floor') as THREE.Mesh;
    ((floor.material as THREE.ShaderMaterial).uniforms.uX.value = this.xray);
    floor.visible = bodyFade > 0.02;
    this.bloom.strength = 0.25 + 0.3 * this.xray * bodyFade;
    this.onFrame?.(dt);
    if (this.renderEnabled && this.dirty) {
      this.composer.render(dt);
      this.dirty = false;
    }
  }
}
