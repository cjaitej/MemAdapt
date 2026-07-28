"""Mixed-precision selection that survives moving between GPUs.

bfloat16 needs compute capability >= 8.0 (Ampere). That covers the RTX 3050 this was
developed on, but *not* the free cloud GPUs:

    Tesla T4    Turing, cc 7.5  -> fp16 tensor cores, no bf16
    Tesla P100  Pascal, cc 6.0  -> fp16 arithmetic, no tensor cores, no bf16
    RTX 3050    Ampere, cc 8.6  -> bf16

Hardcoding bf16 therefore breaks silently-ish on Kaggle/Colab: PyTorch either raises
or falls back to an emulation path slow enough to look like a hang.

The difference is not just a dtype swap. bf16 keeps fp32's exponent range, so
gradients never underflow and no loss scaling is needed. fp16 has a much narrower
range and **requires a GradScaler**, or small gradients flush to zero and the model
quietly trains worse. `select_precision` returns both the dtype and whether a scaler
is needed, so callers cannot get that pairing wrong.
"""

import torch


def select_precision(device_type="cuda", prefer=None, verbose=True):
    """Choose an autocast dtype for this GPU.

    Parameters
    ----------
    prefer : force "bf16", "fp16" or "fp32"; None auto-detects.

    Returns
    -------
    (dtype, use_scaler, label)
    """
    if device_type != "cuda" or not torch.cuda.is_available():
        return torch.float32, False, "fp32 (cpu)"

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    bf16_ok = bf16_supported()

    if prefer == "fp32":
        dtype, scaler, label = torch.float32, False, "fp32 (forced)"
    elif prefer == "fp16":
        dtype, scaler, label = torch.float16, True, "fp16 (forced)"
    elif prefer == "bf16":
        if not bf16_ok:
            raise RuntimeError(
                f"bf16 requested but {name} (cc {cap[0]}.{cap[1]}) does not support it. "
                "bf16 needs compute capability >= 8.0. Use --precision fp16."
            )
        dtype, scaler, label = torch.bfloat16, False, "bf16 (forced)"
    elif bf16_ok:
        dtype, scaler, label = torch.bfloat16, False, "bf16"
    else:
        # fp16 + loss scaling. Slower to converge than bf16 in pathological cases but
        # correct, and far better than fp32 on a T4's tensor cores.
        dtype, scaler, label = torch.float16, True, "fp16 + GradScaler"

    if verbose:
        print(f"precision   : {label} on {name} (cc {cap[0]}.{cap[1]})")
    return dtype, scaler, label


def make_scaler(use_scaler):
    """GradScaler when fp16 needs one, otherwise a no-op with the same interface.

    Returning a disabled scaler rather than None keeps the training loop free of
    `if scaler is not None` branches at every use site.
    """
    return torch.amp.GradScaler("cuda", enabled=bool(use_scaler))


def bf16_supported():
    """True only where bf16 runs on hardware, not via emulation.

    `torch.cuda.is_bf16_supported()` alone is not enough: recent PyTorch returns True
    on pre-Ampere cards because it counts an emulation path. Reporting that as "bf16"
    on a T4 or P100 contradicts what `select_precision` actually picks, so both go
    through the same cc >= 8.0 rule.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()


def describe_device():
    if not torch.cuda.is_available():
        return "cpu"
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return (f"{torch.cuda.get_device_name(0)} | cc {cap[0]}.{cap[1]} | "
            f"{total:.1f} GB | bf16={bf16_supported()} | "
            f"gpus={torch.cuda.device_count()}")
