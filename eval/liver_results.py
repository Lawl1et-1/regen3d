"""Training results as 3D images (no charts).

    python eval/liver_results.py            # final model + every saved checkpoint

Writes figures/liver/:
  01_repair.png       a swipe incision and its repair, with measured shape / function / vessels
  02_vessels.png      X-ray of the vascular tree: healthy, cut, repaired
  03_checkpoints.png  the same incision repaired by the model at each training stage
  results.json        all the numbers shown on the images
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))

from damage import wedge_region  # noqa: E402
from metrics import all_metrics, classify  # noqa: E402
from nca import CHANNELS, NCA3D  # noqa: E402
from render3d import _dev, render  # noqa: E402

OUT = ROOT / "figures" / "liver"
SETTLE = 60   # steps the AI runs on the healthy organ before the cut
REPAIR = 300
TEXT = (231, 236, 243)
MUTED = (139, 149, 166)
CYAN = (95, 225, 255)


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_model(path: Path, dev):
    ck = torch.load(path, map_location=dev)
    m = NCA3D(hidden=ck.get("hidden", 96)).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["iter"] + 1


def swipe_lesion(labels: np.ndarray, depth_frac=0.6, w0=2.2) -> np.ndarray:
    """A user-like swipe from the front: diagonal stroke across the organ, cutting in along -z."""
    n = labels.shape[0]
    alive = labels > 0
    zz, yy, xx = np.nonzero(alive)
    cx, cy = xx.mean(), yy.mean()
    col = alive[:, int(round(cy)), int(round(cx))]
    zf = np.nonzero(col)[0]
    e = np.array([cx, cy, zf.max()], float)               # front surface at the centre
    v = np.array([0.0, 0.0, -1.0])                         # into the tissue
    u = np.array([0.55, -0.83, 0.0]); u /= np.linalg.norm(u)   # top-left -> bottom-right on screen
    nrm = np.cross(u, v)
    thick = zf.max() - zf.min() + 1
    pts = np.stack([xx, yy, zz], 1).astype(float)
    band = np.abs((pts - e) @ nrm) < 1.5
    s = (pts[band] - e) @ u
    m = wedge_region(n, e.tolist(), nrm.tolist(), v.tolist(), u.tolist(),
                     float(s.min()) - 1, float(s.max()) + 1, float(thick * depth_frac), w0)
    return m.numpy()


@torch.no_grad()
def run(model, start, mask_np, dev, frames_at=()):
    """Healthy organ -> AI runs a while -> swipe incision -> AI repairs."""
    torch.manual_seed(0)
    x = start.clone()
    for _ in range(SETTLE):
        x = model.step(x)
    healthy = x[0].cpu()
    y = x * (~torch.from_numpy(mask_np).to(dev)).float()
    cut = y[0].cpu()
    seq = []
    for k in range(REPAIR + 1):
        if k in frames_at:
            seq.append((k, y[0].cpu()))
        if k < REPAIR:
            y = model.step(y)
    return healthy, cut, seq


def wound_of(cut_state: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    alive = cut_state[0].numpy() > 0.1
    return torch.from_numpy(ndimage.binary_dilation(mask, iterations=2) & alive)


def panel(img: np.ndarray, title: str, lines=(), accent=None) -> Image.Image:
    h, w = img.shape[:2]
    pad = 64 + 20 * len(lines)
    out = Image.new("RGB", (w, h + pad), (5, 7, 11))
    out.paste(Image.fromarray((img * 255).astype(np.uint8)), (0, 0))
    d = ImageDraw.Draw(out)
    d.text((12, h + 10), title, fill=accent or TEXT, font=font(17, True))
    for i, ln in enumerate(lines):
        d.text((12, h + 38 + 20 * i), ln, fill=MUTED, font=font(14))
    return out


def sheet(panels, title: str, subtitle: str, cols=None) -> Image.Image:
    cols = cols or len(panels)
    rows = (len(panels) + cols - 1) // cols
    pw = max(p.width for p in panels); ph = max(p.height for p in panels)
    gap, top = 8, 78
    out = Image.new("RGB", (cols * pw + (cols + 1) * gap, top + rows * (ph + gap) + 30), (5, 7, 11))
    d = ImageDraw.Draw(out)
    d.text((gap + 4, 14), title, fill=TEXT, font=font(26, True))
    d.text((gap + 4, 48), subtitle, fill=MUTED, font=font(15))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        out.paste(p, (gap + c * (pw + gap), top + r * (ph + gap)))
    d.text((gap + 4, out.height - 24), "REGEN 3D · simulation of a learned model — not a medical tool",
           fill=(86, 96, 112), font=font(13))
    return out


def fmt(m):
    return [f"shape {m['iou'] * 100:.1f}%  ·  function {m['function'] * 100:.1f}%",
            f"vessels linked to hilum {m['connectivity'] * 100:.1f}%"]


def main():
    dev = _dev()
    run_dir = ROOT / "train" / "runs" / "liver"
    target = np.load(ROOT / "targets" / "data" / "liver.npy")
    meta = json.loads((ROOT / "targets" / "data" / "liver.json").read_text())
    hil = meta["hilum"]
    n = meta["n"]
    mask = swipe_lesion(target)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"lesion_voxels": int((mask & (target > 0)).sum()),
               "lesion_fraction": float((mask & (target > 0)).sum() / (target > 0).sum())}

    # ---------------------------------------------------------------- final model
    model, it = load_model(run_dir / "model.pt", dev)
    tgt_state = torch.zeros(CHANNELS, n, n, n)
    tgt_state[0] = torch.from_numpy(target > 0).float()
    for t in range(1, 5):
        tgt_state[t] = torch.from_numpy(target == t).float()
    start = tgt_state[None].to(dev)  # the healthy organ, exactly as the viewer starts it
    times = (0, 10, 25, 50, 100, 200, 300)
    healthy, cut, seq = run(model, start, mask, dev, times)
    m_h = all_metrics(healthy.numpy(), target, hil, surface=True)
    m_c = all_metrics(cut.numpy(), target, hil)
    results["final"] = dict(iteration=it, healthy=m_h, lesion=m_c, repair={})

    wnd = wound_of(cut, mask)
    r_panels = [panel(render(healthy), "healthy", fmt(m_h)),
                panel(render(cut, wound=wnd), "swipe incision", fmt(m_c) + [f"{results['lesion_fraction'] * 100:.1f}% of the tissue removed"])]
    for k, s in seq[1:]:
        m = all_metrics(s.numpy(), target, hil)
        results["final"]["repair"][k] = m
        regrown = torch.from_numpy(mask) & (s[0] > 0.1)
        r_panels.append(panel(render(s, regrown=regrown), f"+{k} steps", fmt(m), accent=CYAN if k == times[-1] else None))
    sheet(r_panels, "The AI repairs a swipe incision",
          "No repair rule is written anywhere: the same network learned it from thousands of random cuts.  Regrown tissue tinted cyan.",
          cols=4).save(OUT / "01_repair.png")

    final = seq[-1][1]
    v_panels = [panel(render(healthy, mode="xray"), "healthy vessels", [f"linked to hilum {m_h['connectivity'] * 100:.1f}%"]),
                panel(render(cut, mode="xray"), "after the incision", [f"linked to hilum {m_c['connectivity'] * 100:.1f}%"]),
                panel(render(final, mode="xray"), f"after {REPAIR} steps", [f"linked to hilum {results['final']['repair'][REPAIR]['connectivity'] * 100:.1f}%"], accent=CYAN)]
    sheet(v_panels, "Vessels reconnect", "X-ray view: portal tree (red) and hepatic veins (blue).").save(OUT / "02_vessels.png")

    # ---------------------------------------------------------------- checkpoints
    rows = []
    results["checkpoints"] = []
    for ck in sorted(run_dir.glob("ckpt_*.pt")):
        mdl, it_c = load_model(ck, dev)
        h_c, c_c, s_c = run(mdl, start, mask, dev, (REPAIR,))
        mh, mr = all_metrics(h_c.numpy(), target, hil), all_metrics(s_c[-1][1].numpy(), target, hil)
        results["checkpoints"].append(dict(iteration=it_c, healthy=mh, repaired=mr))
        regrown = torch.from_numpy(mask) & (s_c[-1][1][0] > 0.1)
        rows += [panel(render(h_c), f"iteration {it_c}: healthy", fmt(mh)),
                 panel(render(c_c, wound=wound_of(c_c, mask)), "same incision", []),
                 panel(render(s_c[-1][1], regrown=regrown), f"after {REPAIR} steps", fmt(mr), accent=CYAN)]
    if rows:
        sheet(rows, "Watch the AI learn", "The same incision, repaired by the model saved at each stage of training.",
              cols=3).save(OUT / "03_checkpoints.png")
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "final"}, indent=1)[:2000])
    print("final repair", {k: round(v["iou"], 3) for k, v in results["final"]["repair"].items()})


if __name__ == "__main__":
    main()
