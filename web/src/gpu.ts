// WebGPU initialisation with a clear message when it is unavailable (never a silent failure).
export interface Gpu {
  device: GPUDevice;
  ctx: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
}

export function fatal(title: string, body: string) {
  const ui = document.getElementById('ui')!;
  ui.innerHTML = `<div class="fatal"><div class="box glass"><h1>${title}</h1>${body}</div></div>`;
}

export async function initGPU(canvas: HTMLCanvasElement): Promise<Gpu | null> {
  if (!('gpu' in navigator)) {
    fatal('WebGPU is not available',
      `<p>REGEN 3D runs the cellular automaton and the volume renderer on your GPU with WebGPU.</p>
       <p>Use a recent Chrome, Edge or Opera (version 113+) on desktop, or Safari 26+. On Linux, enable
       <code>chrome://flags/#enable-unsafe-webgpu</code>. Firefox needs <code>dom.webgpu.enabled</code>.</p>`);
    return null;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    fatal('No compatible GPU adapter', '<p>WebGPU is present but no adapter was found. Your GPU may be blocklisted, or hardware acceleration is disabled in the browser settings.</p>');
    return null;
  }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
  });
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') fatal('The GPU device was lost', `<p>${info.message || 'The browser reset the GPU.'}</p><p>Reload the page to restart.</p>`);
  });
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message));
  const ctx = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  return { device, ctx, format, canvas };
}
