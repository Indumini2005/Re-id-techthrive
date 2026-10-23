"""Checkpoint save/load shared by training, export and evaluation.

Kept in one place so the three of them cannot disagree about what a
checkpoint contains - in particular the architecture, which has to survive
the trip from training to ONNX export or a width-16 'nano' model silently
gets loaded into a width-32 graph.

PyTorch 2.6 changed ``torch.load`` to default ``weights_only=True``, which
refuses any pickled object that is not a tensor. A checkpoint that stores
argparse values verbatim therefore contains ``pathlib.WindowsPath`` objects
and will not load at all. Everything non-tensor is converted to a plain
string on the way in, and the loader falls back for checkpoints written
before that fix.
"""

from pathlib import Path

import torch


def _plain(value):
    """Reduce an arbitrary argparse value to something weights_only-safe."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_checkpoint(path, model, ema, optimizer, epoch, step, best_psnr, args):
    """Write a checkpoint that ``torch.load`` can read with weights_only=True."""
    payload = {
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict() if ema else None,
        "ema_step": getattr(ema, "step", 0) if ema else 0,
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "best_psnr": float(best_psnr),
        "arch": {
            "width": int(args.width),
            "enc_blks": [int(v) for v in args.enc_blks],
            "middle_blks": int(args.middle_blks),
            "dec_blks": [int(v) for v in args.dec_blks],
            "name": _plain(getattr(args, "arch", None)),
        },
        "args": {k: _plain(v) for k, v in vars(args).items()},
    }
    torch.save(payload, str(path))


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint, preferring the safe path."""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception:
        # Written before the _plain() fix above: it holds pickled Path
        # objects. These are the user's own training runs, not untrusted
        # downloads, so loading them fully is reasonable.
        print("note: %s predates the weights-only fix; loading it fully."
              % Path(path).name)
        return torch.load(str(path), map_location=map_location, weights_only=False)


def build_from_checkpoint(ckpt, use_ema=True):
    """Reconstruct the model described by a checkpoint and load its weights."""
    from training.nafnet import build_nafnet

    arch = ckpt.get("arch", {})
    model = build_nafnet(
        width=arch.get("width"),
        enc=arch.get("enc_blks"),
        middle=arch.get("middle_blks"),
        dec=arch.get("dec_blks"),
    )
    state = ckpt.get("ema") if (use_ema and ckpt.get("ema")) else ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, arch
