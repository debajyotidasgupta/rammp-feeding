#!/usr/bin/env python3
"""Save an INT8 copy of LocateAnything, then verify it before anything is deleted.

INT8 is the deployment target (docs/perception-quantization.md), so the 7.7 GB
of BF16 safetensors is dead weight on a box with 6 GB free. This writes an INT8
state_dict, reconstructs the model from config alone, and runs a real detection
through the reconstruction.

The order matters and is not negotiable: save, reload, VERIFY, and only then
report that the originals are safe to remove. A quantised checkpoint that cannot
be loaded back is worse than no space saving, and torchao checkpoints are
version-coupled -- they carry tensor subclasses, so they load only under a
compatible torchao.

    python -m feeding.quantize_and_save --out ~/models/la-int8.pt
    python -m feeding.quantize_and_save --out ~/models/la-int8.pt --verify-only
"""
from __future__ import annotations

import argparse
import pathlib
import time

import torch
from PIL import Image

MODEL_ID = "nvidia/LocateAnything-3B"


def build_empty(model_id: str, dtype):
    """Model skeleton from config only -- no big weight files touched."""
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModel.from_config(cfg, trust_remote_code=True)
    return model.to_empty(device="cpu").to(dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path.home() / "models" / "la-int8.pt"))
    ap.add_argument("--image", default="/tmp/probe_frame.jpg")
    ap.add_argument("--phrase", default="face")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(a.image).convert("RGB")

    from feeding.locate_anything import LocateAnything, parse_boxes
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    if not a.verify_only:
        print("loading BF16 ...")
        la = LocateAnything()
        ref = la.locate_one(img, a.phrase)
        print("  reference box: %s" % (ref and (round(ref.centre[0]), round(ref.centre[1]))))
        print("quantising to INT8 ...")
        quantize_(la.model, Int8WeightOnlyConfig())
        t0 = time.perf_counter()
        torch.save(la.model.state_dict(), out)
        gb = out.stat().st_size / 1e9
        print("  wrote %s  (%.2f GB, %.0f s)" % (out, gb, time.perf_counter() - t0))
        del la
        torch.cuda.empty_cache()

    print("verifying: rebuilding from config + INT8 checkpoint only ...")
    model = build_empty(MODEL_ID, torch.bfloat16)
    quantize_(model, Int8WeightOnlyConfig())          # match the saved subclasses
    sd = torch.load(out, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("  missing %d, unexpected %d" % (len(missing), len(unexpected)))
    model = model.to("cuda").eval()

    from transformers import AutoTokenizer, AutoProcessor
    la2 = LocateAnything.__new__(LocateAnything)
    la2.torch, la2.dtype, la2.device = torch, torch.bfloat16, "cuda"
    la2.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    la2.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    la2.model = model
    box = la2.locate_one(img, a.phrase)
    print("  reloaded box : %s" % (box and (round(box.centre[0]), round(box.centre[1]))))
    if box is None:
        print("\n  VERIFY FAILED -- do NOT delete the BF16 weights.")
        return 1
    print("\n  VERIFY OK. The BF16 safetensors blobs can now be removed;")
    print("  keep config/*.json and the *.py modelling files, they are tiny and required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
