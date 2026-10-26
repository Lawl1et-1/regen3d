// Orbit camera with damping, zoom, pan, idle auto-rotate and scripted orbits.
import { damp, invert, lookAt, mul, perspective, type M4, type V3, v3, clamp } from './math';

export class OrbitCamera {
  yaw = 0.7;
  pitch = 0.25;
  dist = 3.4;
  target: V3 = [0, 0, 0];
  private gYaw = this.yaw;
  private gPitch = this.pitch;
  private gDist = this.dist;
  private gTarget: V3 = [0, 0, 0];
  idle = 0;
  autoRotate = true;
  autoSpeed = 0.12;
  fov = 0.62;
  shake = 0;
  view: M4 = new Float32Array(16);
  proj: M4 = new Float32Array(16);
  viewProj: M4 = new Float32Array(16);
  invViewProj: M4 = new Float32Array(16);
  eye: V3 = [0, 0, 3];

  orbit(dx: number, dy: number) {
    this.gYaw -= dx * 0.006;
    this.gPitch = clamp(this.gPitch + dy * 0.005, -1.45, 1.45);
    this.idle = 0;
  }
  zoom(f: number) { this.gDist = clamp(this.gDist * f, 1.4, 9); this.idle = 0; }
  pan(dx: number, dy: number) {
    const r: V3 = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
    const u: V3 = [0, 1, 0];
    const s = this.dist * 0.0012;
    this.gTarget = v3.add(this.gTarget, v3.add(v3.scale(r, -dx * s), v3.scale(u, dy * s)));
    this.idle = 0;
  }
  setGoal(yaw: number, pitch: number, dist: number, target?: V3) {
    this.gYaw = yaw; this.gPitch = pitch; this.gDist = dist;
    if (target) this.gTarget = target;
  }
  /** Rotate the goal yaw continuously (cinematic / forecast orbits). */
  spin(rate: number, dt: number) { this.gYaw += rate * dt; }
  get goalYaw() { return this.gYaw; }
  get goalTarget() { return this.gTarget; }
  reset() { this.setGoal(0.7, 0.25, 3.4, [0, 0, 0]); }

  update(dt: number, aspect: number, time: number) {
    this.idle += dt;
    if (this.autoRotate && this.idle > 4) this.gYaw += this.autoSpeed * dt * Math.min(1, (this.idle - 4) / 2);
    this.yaw = damp(this.yaw, this.gYaw, 7, dt);
    this.pitch = damp(this.pitch, this.gPitch, 7, dt);
    this.dist = damp(this.dist, this.gDist, 6, dt);
    this.target = [0, 1, 2].map((i) => damp(this.target[i], this.gTarget[i], 6, dt)) as V3;
    this.shake = Math.max(0, this.shake - dt * 3);
    const cp = Math.cos(this.pitch);
    let eye: V3 = v3.add(this.target, [Math.sin(this.yaw) * cp * this.dist, Math.sin(this.pitch) * this.dist, Math.cos(this.yaw) * cp * this.dist]);
    let tgt = this.target;
    if (this.shake > 0) {
      const s = this.shake * this.shake * 0.04;
      const j: V3 = [Math.sin(time * 91) * s, Math.cos(time * 77) * s, Math.sin(time * 63) * s];
      eye = v3.add(eye, j); tgt = v3.add(tgt, j);
    }
    this.eye = eye;
    this.view = lookAt(eye, tgt, [0, 1, 0]);
    this.proj = perspective(this.fov, aspect, 0.05, 50);
    this.viewProj = mul(this.proj, this.view);
    this.invViewProj = invert(this.viewProj);
  }

  /** World-space ray through a pixel (ndc in [-1, 1]). */
  ray(ndcX: number, ndcY: number): { o: V3; d: V3 } {
    const p = (z: number): V3 => {
      const m = this.invViewProj;
      const x = ndcX, y = ndcY;
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w, (m[1] * x + m[5] * y + m[9] * z + m[13]) / w, (m[2] * x + m[6] * y + m[10] * z + m[14]) / w];
    };
    const a = p(0), b = p(1);
    return { o: this.eye, d: v3.norm(v3.sub(b, a)) };
  }

  project(p: V3): [number, number, number] {
    const m = this.viewProj;
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    return [(m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w, (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w, w];
  }
}
