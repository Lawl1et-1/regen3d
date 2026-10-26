import type { App, Controller } from './app';

export class BodyView implements Controller {
  constructor(private app: App) {}
  mount() { location.hash = `#/lab/${this.app.organs[0]?.name ?? 'blob'}`; }
  unmount() {}
  frame() {}
}
