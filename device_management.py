"""
device_management.py
=======================

Centralized hardware separation between training and inference, per the
requirement: the classical inference latency (tau_inf) that feeds the
quantum memory's decoherence channel must reflect what an edge
microprocessor would actually experience -- NOT the latency of a GPU call
(which is dominated by PCIe bus transfer overhead for a single sample and
would badly underrepresent real edge-deployment latency).

    TRAIN_DEVICE      : GPU if available (cuda), else CPU.
                        Used ONLY during model.fit() / training loops, where
                        batched throughput matters and PCIe overhead is
                        amortized across the whole batch.
    INFERENCE_DEVICE  : ALWAYS CPU, regardless of what's available.
                        Used for every latency-honest benchmarking loop:
                        model.eval() + torch.no_grad() + batch_size=1,
                        exactly mimicking a single WDM telemetry sample
                        arriving at an edge repeater's onboard microprocessor.
"""

import torch

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE_DEVICE = torch.device("cpu")


def move_to_train_device(*tensors_or_modules):
    """Moves any number of tensors/modules to TRAIN_DEVICE, returned in the same order."""
    moved = [t.to(TRAIN_DEVICE) for t in tensors_or_modules]
    return moved[0] if len(moved) == 1 else moved


def prepare_for_honest_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Moves a (possibly GPU-trained) model back to CPU and puts it in eval
    mode. Call this ONCE after training, before entering the batch-size=1
    CPU benchmarking loop -- never re-move the model mid-loop (that would
    reintroduce transfer overhead into the very latency we're trying to
    measure honestly).
    """
    model = model.to(INFERENCE_DEVICE)
    model.eval()
    return model


def report_devices() -> None:
    print(f"TRAIN_DEVICE (fit phase, batched):            {TRAIN_DEVICE}")
    print(f"INFERENCE_DEVICE (benchmark phase, batch=1):  {INFERENCE_DEVICE}  (always CPU, by design)")
