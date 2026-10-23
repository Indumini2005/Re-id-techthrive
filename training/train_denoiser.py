"""Train the stage-1 NAFNet denoiser.

Typical run (synthetic pairs from a folder of sharp images):

    python -m training.train_denoiser --train_dir data/clean --epochs 200 \
        --patch_size 256 --batch_size 8 --preset cctv

With real paired data:

    python -m training.train_denoiser --train_dir data/clean \
        --noisy_train_dir data/noisy --val_dir data/val_clean

Resume:

    python -m training.train_denoiser --train_dir data/clean --resume runs/nafnet/last.pth

Checkpoints land in ``runs/<name>/``: ``last.pth``, ``best.pth`` (highest
validation PSNR) and periodic sample grids so you can see what it is doing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from training.dataset import build_loaders, to_image  # noqa: E402
from training.losses import build_loss, psnr, ssim  # noqa: E402
from training.nafnet import build_nafnet, count_parameters  # noqa: E402


# --- helpers ---------------------------------------------------------------
class EMA:
    """Exponential moving average of the weights. The averaged model is what
    you export - it is reliably 0.1-0.3 dB better than the raw weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model: nn.Module, loader, device: torch.device, amp: bool, max_batches: int = 0):
    model.eval()
    psnr_sum = ssim_sum = 0.0
    count = 0
    for i, (noisy, clean) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = model(noisy)
        pred = pred.float().clamp(0, 1)
        psnr_sum += psnr(pred, clean)
        ssim_sum += ssim(pred, clean)
        count += 1
    model.train()
    if count == 0:
        return 0.0, 0.0
    return psnr_sum / count, ssim_sum / count


@torch.no_grad()
def dump_samples(model: nn.Module, loader, device: torch.device, out_path: Path, n: int = 4):
    """Writes a noisy | denoised | clean strip so you can eyeball progress."""
    import cv2

    model.eval()
    rows = []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy).float().clamp(0, 1)
        for i in range(min(n - len(rows), noisy.shape[0])):
            rows.append(np.hstack([to_image(noisy[i]), to_image(pred[i]), to_image(clean[i])]))
        if len(rows) >= n:
            break
    model.train()
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), np.vstack(rows))


def save_checkpoint(path: Path, model, ema, optimizer, epoch, step, best_psnr, args):
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_psnr": best_psnr,
            "arch": {
                "width": args.width,
                "enc_blks": args.enc_blks,
                "middle_blks": args.middle_blks,
                "dec_blks": args.dec_blks,
            },
            "args": vars(args),
        },
        path,
    )


# --- main ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train NAFNet for CCTV denoising")

    # data
    p.add_argument("--train_dir", type=Path, required=True, help="folder of CLEAN images")
    p.add_argument("--noisy_train_dir", type=Path, default=None, help="paired noisy images (optional)")
    p.add_argument("--val_dir", type=Path, default=None)
    p.add_argument("--noisy_val_dir", type=Path, default=None)
    p.add_argument("--preset", default="cctv", choices=["light", "cctv", "harsh"])
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--repeat", type=int, default=1, help="patches sampled per image per epoch")
    p.add_argument("--val_split", type=float, default=0.05)

    # model
    p.add_argument("--arch", default=None, choices=sorted(config.NAFNET_ARCHS),
                   help="named size preset; 'nano' for the original Jetson Nano. "
                        "Individual --width/--enc_blks flags override it.")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--enc_blks", type=int, nargs="+", default=None)
    p.add_argument("--middle_blks", type=int, default=None)
    p.add_argument("--dec_blks", type=int, nargs="+", default=None)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.9])
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", default="combined", choices=["psnr", "charbonnier", "combined"])
    p.add_argument("--edge_weight", type=float, default=0.05)
    p.add_argument("--ema_decay", type=float, default=0.999, help="0 disables EMA")

    # runtime
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="nafnet")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--val_every", type=int, default=1, help="epochs between validations")
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=1)
    return p.parse_args()


def resolve_arch(args):
    """Fill in width/depth from the named preset, honouring explicit flags.

    The resolved values are written back onto ``args`` so they land in the
    checkpoint, which is what export and inference read. Keeping the shape in
    one place is what stops a width-16 'nano' checkpoint from being loaded
    into a width-32 graph months later.
    """
    preset = config.NAFNET_ARCHS[args.arch] if args.arch else {
        "width": config.NAFNET_WIDTH,
        "enc": config.NAFNET_ENC_BLKS,
        "middle": config.NAFNET_MIDDLE_BLKS,
        "dec": config.NAFNET_DEC_BLKS,
    }
    if args.width is None:
        args.width = preset["width"]
    if args.enc_blks is None:
        args.enc_blks = list(preset["enc"])
    if args.middle_blks is None:
        args.middle_blks = preset["middle"]
    if args.dec_blks is None:
        args.dec_blks = list(preset["dec"])

    if len(args.enc_blks) != len(args.dec_blks):
        raise SystemExit(
            "encoder and decoder depths must match (%d vs %d)"
            % (len(args.enc_blks), len(args.dec_blks))
        )
    return args


def main():
    args = parse_args()
    args = resolve_arch(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    out_dir = config.RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w", encoding="utf-8") as fh:
        json.dump({k: str(v) for k, v in vars(args).items()}, fh, indent=2)

    # -- data
    train_loader, val_loader = build_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        noisy_train_dir=args.noisy_train_dir,
        noisy_val_dir=args.noisy_val_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        workers=args.workers,
        preset=args.preset,
        val_split=args.val_split,
        repeat=args.repeat,
    )

    # -- model
    model = build_nafnet(
        width=args.width, enc=args.enc_blks, middle=args.middle_blks,
        dec=args.dec_blks, drop_out_rate=args.dropout,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = build_loss(args.loss, edge_weight=args.edge_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay
    )
    # torch.cuda.amp.GradScaler is deprecated from 2.4 onward.
    use_amp = args.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    start_epoch, global_step, best_psnr = 0, 0, -1.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ema and ckpt.get("ema"):
            ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("step", 0)
        best_psnr = ckpt.get("best_psnr", -1.0)
        print(f"resumed from {args.resume} @ epoch {start_epoch} (best PSNR {best_psnr:.3f})")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs

    print("-" * 66)
    print(f"model      : NAFNet w{args.width} enc{args.enc_blks} mid{args.middle_blks} "
          f"dec{args.dec_blks}" + (f"  (--arch {args.arch})" if args.arch else ""))
    print(f"params     : {count_parameters(model) / 1e6:.2f}M")
    print(f"device     : {device} | amp={args.amp} | ema={args.ema_decay}")
    print(f"train/val  : {len(train_loader.dataset)} / {len(val_loader.dataset)} samples")
    print(f"schedule   : {args.epochs} epochs x {steps_per_epoch} steps = {total_steps}")
    print(f"output     : {out_dir}")
    print("-" * 66)

    log_path = out_dir / "log.csv"
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    if new_log:
        logger.writerow(["epoch", "step", "lr", "train_loss", "val_psnr", "val_ssim", "secs"])

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_start = time.time()
            running_loss, seen = 0.0, 0

            for noisy, clean in train_loader:
                lr = cosine_lr(global_step, total_steps, args.lr, args.warmup_steps, args.min_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                if device.type == "cuda":
                    noisy = noisy.to(memory_format=torch.channels_last)
                    clean = clean.to(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    pred = model(noisy)
                    loss, parts = criterion(pred.float(), clean)

                if not torch.isfinite(loss):
                    print(f"  [step {global_step}] non-finite loss, batch skipped")
                    global_step += 1
                    continue

                scaler.scale(loss).backward()
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                if ema:
                    ema.update(model)

                running_loss += parts["total"]
                seen += 1
                global_step += 1

                if seen % 50 == 0:
                    print(
                        f"  epoch {epoch+1:3d}/{args.epochs} "
                        f"step {seen:5d}/{steps_per_epoch} "
                        f"lr {lr:.2e} loss {running_loss / seen:+.4f}",
                        flush=True,
                    )

            train_loss = running_loss / max(1, seen)
            eval_model = ema.shadow if ema else model

            val_psnr = val_ssim = float("nan")
            if (epoch + 1) % args.val_every == 0:
                val_psnr, val_ssim = validate(eval_model, val_loader, device, args.amp)
                print(
                    f"epoch {epoch+1:3d} | loss {train_loss:+.4f} | "
                    f"val PSNR {val_psnr:.3f} dB | SSIM {val_ssim:.4f} | "
                    f"{time.time() - epoch_start:.1f}s"
                )
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save_checkpoint(out_dir / "best.pth", model, ema, optimizer,
                                    epoch, global_step, best_psnr, args)
                    print(f"           -> new best ({best_psnr:.3f} dB) saved")

            if (epoch + 1) % args.sample_every == 0:
                dump_samples(eval_model, val_loader, device, out_dir / "samples" / f"epoch_{epoch+1:04d}.png")

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(out_dir / "last.pth", model, ema, optimizer,
                                epoch, global_step, best_psnr, args)

            logger.writerow([epoch + 1, global_step, f"{lr:.3e}", f"{train_loss:.5f}",
                             f"{val_psnr:.4f}", f"{val_ssim:.5f}", f"{time.time() - epoch_start:.1f}"])
            log_file.flush()

    except KeyboardInterrupt:
        print("\ninterrupted - saving last.pth")
        save_checkpoint(out_dir / "last.pth", model, ema, optimizer,
                        epoch, global_step, best_psnr, args)
    finally:
        log_file.close()

    print("-" * 66)
    print(f"done. best val PSNR {best_psnr:.3f} dB")
    print(f"export it with:  python export/export_onnx.py --ckpt {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
