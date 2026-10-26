"""Train a 3D NCA to grow and regenerate a procedural organ.

    python train/train.py --organ blob --iters 1500
    python train/train.py --organ liver --iters 6000 --amp --ckpt

Sample-pool training (Mordvintsev et al. 2020) in 3D with spherical holes,
planar half-space cuts and tunnels applied to the best samples of each batch.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from damage import DAMAGES, random_damage  # noqa: E402
from nca import CHANNELS, NCA3D, make_seed  # noqa: E402
from snapshots import render, save_png, strip  # noqa: E402


def load_target(organ: str, device):
    labels = np.load(ROOT / "targets" / "data" / f"{organ}.npy")
    meta = json.loads((ROOT / "targets" / "data" / f"{organ}.json").read_text())
    lab = torch.from_numpy(labels.astype(np.int64)).to(device)
    tgt = torch.zeros(5, *labels.shape, device=device)
    tgt[0] = (lab > 0).float()
    for t in range(1, 5):
        tgt[t] = (lab == t).float()
    weight = torch.ones_like(tgt[0])
    weight[lab == 1] = 2.0
    weight[lab == 3] = 4.0
    weight[lab == 4] = 3.0
    return tgt, weight, meta


def per_sample_loss(y, tgt, weight):
    err = (y[:, :5] - tgt[None]) ** 2 * weight[None, None]
    return err.mean(dim=(1, 2, 3, 4))


def iou(state, tgt):
    a = state[0] > 0.1
    b = tgt[0] > 0.5
    return ((a & b).sum() / ((a | b).sum() + 1e-6)).item()


@torch.no_grad()
def log_snapshot(model, seed, tgt, out: Path, it: int, n_grow: int):
    model.eval()
    x = seed.clone()
    frames, ks = [], []
    marks = sorted({0, n_grow // 8, n_grow // 4, n_grow // 2, n_grow, int(n_grow * 1.5)})
    for k in range(marks[-1] + 1):
        if k in marks:
            frames.append(render(x[0]))
            ks.append(k)
        x = model.step(x)
    grown = x.clone()
    # regeneration test: a deep swipe incision, then let it heal
    torch.manual_seed(it)
    regen = [render(grown[0])]
    y = DAMAGES["wedge"](grown[0].clone(), (0.5, 0.6))[None]
    regen.append(render(y[0]))
    for k in range(1, n_grow + 1):
        y = model.step(y)
        if k in (n_grow // 4, n_grow // 2, n_grow):
            regen.append(render(y[0]))
    target_img = render(torch.cat([tgt, torch.zeros(CHANNELS - 5, *tgt.shape[1:], device=tgt.device)]))
    top = strip(frames + [target_img])
    bot = strip(regen)
    w = max(top.shape[1], bot.shape[1])
    pad = lambda im: np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)))  # noqa: E731
    img = np.concatenate([pad(top), np.zeros((6, w, 3)), pad(bot)], 0)
    save_png(img, out / f"snap_{it:05d}.png")
    model.train()
    return iou(grown[0], tgt), iou(y[0], tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", default="blob")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--steps", type=int, nargs=2, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--damage-start", type=int, default=300)
    ap.add_argument("--n-damage", type=int, default=2)
    ap.add_argument("--snap-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--vram", type=float, default=0.6,
                    help="max share of GPU memory for training, so the browser/desktop keep enough")
    ap.add_argument("--damage", nargs="+", default=["wedge:0.8", "sphere:0.2"],
                    help="damage kinds with weights, e.g. wedge:0.8 sphere:0.2")
    ap.add_argument("--init", choices=["seed", "target"], default="target",
                    help="start samples from a single seed cell (grow) or from the healthy organ (repair)")
    ap.add_argument("--checkpoints", type=float, nargs="+", default=[0.08, 0.25, 0.5, 0.75, 1.0],
                    help="fractions of training at which to keep a checkpoint (for 'Watch the AI learn')")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.vram)
    tgt, weight, meta = load_target(args.organ, dev)
    n = meta["n"]
    if args.init == "target":
        steps = args.steps or [40, 72]   # repair is local: no need to cross the whole organ
    else:
        steps = args.steps or ([48, 80] if n <= 32 else [64, 104])
    out = ROOT / "train" / "runs" / args.organ
    out.mkdir(parents=True, exist_ok=True)

    model = NCA3D(hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(args.iters * 0.4), int(args.iters * 0.75)], gamma=0.3)
    start = 0
    if args.resume and (out / "model.pt").exists():
        ck = torch.load(out / "model.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start = ck["iter"] + 1
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    kinds = tuple(d.split(":")[0] for d in args.damage)
    weights = tuple(float(d.split(":")[1]) for d in args.damage)
    keep = sorted({max(1, round(f * args.iters)) for f in args.checkpoints})
    if args.init == "target":
        # the healthy organ exactly as the viewer shows it: alive, one-hot cell types, hidden at 0
        seed = torch.zeros(1, CHANNELS, n, n, n, device=dev)
        seed[0, :5] = tgt
    else:
        seed = make_seed(n, meta["seed"], device=dev)
    pool = seed.cpu().repeat(args.pool, 1, 1, 1, 1)  # kept in system RAM to leave GPU memory free
    log = open(out / "log.csv", "a")
    t0 = time.time()
    for it in range(start, args.iters):
        idx = torch.randperm(args.pool)[: args.batch]
        x = pool[idx].to(dev)
        with torch.no_grad():
            order = per_sample_loss(x, tgt, weight).argsort(descending=True)
            x = x[order]
            idx = idx[order.cpu()]
            x[:1] = seed
            if it >= args.damage_start:
                for j in range(1, 1 + args.n_damage):
                    k = args.batch - j
                    if x[k, 0].max() > 0.1:
                        x[k], _ = random_damage(x[k], kinds, weights)
        k_steps = int(torch.randint(steps[0], steps[1] + 1, (1,)))
        opt.zero_grad(set_to_none=True)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                y = model(x, k_steps, use_checkpoint=args.ckpt)
            loss_s = per_sample_loss(y.float(), tgt, weight)
            overflow = (y - y.clamp(-3.0, 3.0)).abs().mean()
            loss = loss_s.mean() + overflow
            scaler.scale(loss).backward()
        except torch.OutOfMemoryError:
            # stay within the memory cap: skip this batch rather than crash the run
            y = loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"it {it:5d}  skipped (out of memory at {k_steps} steps)", flush=True)
            continue
        scaler.unscale_(opt)
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            pool[idx] = y.detach().float().cpu()
        if it % 20 == 0:
            el = time.time() - t0
            print(f"it {it:5d}  loss {loss.item():.5f}  ovf {overflow.item():.4f}  "
                  f"steps {k_steps}  lr {sched.get_last_lr()[0]:.1e}  {el:6.0f}s", flush=True)
            log.write(f"{it},{loss.item():.6f},{el:.1f}\n")
            log.flush()
        if (it + 1) % args.snap_every == 0 or it == args.iters - 1:
            g, r = log_snapshot(model, seed, tgt, out, it + 1, steps[1] + 16)
            first = "grow" if args.init == "seed" else "stable"
            print(f"  snapshot: {first} IoU {g:.3f}   regen IoU {r:.3f}", flush=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            iter=it, hidden=args.hidden, organ=args.organ), out / "model.pt")
        if it + 1 in keep:
            torch.save(dict(model=model.state_dict(), iter=it, hidden=args.hidden, organ=args.organ),
                       out / f"ckpt_{it + 1:05d}.pt")
            print(f"  checkpoint kept at iteration {it + 1}", flush=True)


if __name__ == "__main__":
    main()
