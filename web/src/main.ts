import './style.css';
import { App } from './app';
import { initGPU } from './gpu';

const canvas = document.getElementById('gpu') as HTMLCanvasElement;
initGPU(canvas).then((gpu) => {
  if (gpu) new App(gpu).start();
});
