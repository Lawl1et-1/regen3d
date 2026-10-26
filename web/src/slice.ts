import type { App, Controller } from './app';
import type { Scene } from './scene';

export class SliceMode implements Controller {
  constructor(private app: App, private scene: Scene) {}
  mount() { location.hash = `#/lab/${this.scene.name}`; void this.app; }
  unmount() {}
  frame() {}
}
