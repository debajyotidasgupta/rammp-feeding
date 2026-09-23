"""Open-vocabulary detection with nvidia/LocateAnything-3B.

Two jobs in the feeding stack, and they have very different duty cycles
(docs/feeding-architecture.md Sec. 2):

  food   -- once per bite cycle, ~every 15 s. Latency barely matters.
  face   -- continuously during TRANSPORT/PRESENT. Latency is the whole game.

Whether one model can do both is an empirical question about this hardware, not
a design decision, so `benchmark()` exists to answer it before anything depends
on the answer. If the face path cannot hit ~10 Hz, food detection still uses
this model and face tracking uses something small; the architecture already
keeps perception out of the control loop, so a slow detector degrades the rate
at which goals are refreshed rather than stalling a trajectory.

Notes that cost time if you rediscover them:

  * Coordinates come back as normalised integers in [0, 1000] inside
    `<box><x1><y1><x2><y2></box>`, NOT pixels. Scale by the image size.
  * `generation_mode="hybrid"` is the default NVIDIA recommends: parallel box
    decoding with an autoregressive fallback when the format comes out
    irregular. "fast" is parallel-only and can truncate.
  * MagiAttention is Hopper/Blackwell only. On Orin we fall back to PyTorch
    SDPA -- functional, slower for the multi-token-prediction path.
  * TensorRT / TensorRT-LLM / Triton are NOT supported by this model yet, so
    the INT4-via-TRT-LLM route does not exist. On a 61 GB unified-memory Orin
    that is fine: BF16 weights are 7.7 GB and quantisation is a latency
    optimisation, not a memory requirement. That is a change from Sec. 7 of the
    architecture, which was written against triphammer's 12 GB card.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

MODEL_ID = "nvidia/LocateAnything-3B"
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# The model tags each category before its boxes:
#   <ref>food pieces</ref><box>..</box><box>..</box><ref>bowl</ref><box>..</box>
# Matching only <box> throws that away and leaves every detection sharing one
# label, which is useless downstream -- the arm has to know which box is food
# and which is the bowl. Walk both tokens in order instead.
TOKEN_RE = re.compile(
    r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def dedupe(boxes: list[Box], iou_thresh: float = 0.5) -> list[Box]:
    """Greedy NMS.

    The model emits the same object many times over: a single query for "face"
    came back with 341 boxes, ~2000 tokens, which is exactly the generation cap.
    Nearly all of them were duplicates of a handful of real detections. Without
    this, box counts are meaningless AND latency is dominated by emitting
    redundant coordinates rather than by the model's actual speed.
    """
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda z: -z.area):
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def parse_boxes(answer: str, width: int, height: int, label: str = "",
                merge: bool = True) -> list[Box]:
    """Model output -> pixel boxes. Coordinates are normalised ints in [0, 1000]."""
    out = []
    for m in BOX_RE.finditer(answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        out.append(Box(x1 / 1000 * width, y1 / 1000 * height,
                       x2 / 1000 * width, y2 / 1000 * height, label))
    return dedupe(out) if merge else out


def parse_labelled(answer: str, width: int, height: int,
                   merge: bool = True) -> list[tuple[str, list[Box]]]:
    """Model output -> [(category, boxes)], in the order the model emitted them.

    Deduping is per category, not globally: a berry sitting inside the bowl
    overlaps the bowl box almost completely, so a global NMS would silently
    drop whichever it sorted second.
    """
    groups: dict[str, list[Box]] = {}
    order: list[str] = []
    current = ""
    for m in TOKEN_RE.finditer(answer):
        ref = m.group(1)
        if ref is not None:
            current = ref.strip()
            if current not in groups:
                groups[current] = []
                order.append(current)
            continue
        x1, y1, x2, y2 = (int(g) for g in m.groups()[1:])
        if current not in groups:
            groups[current] = []
            order.append(current)
        groups[current].append(Box(x1 / 1000 * width, y1 / 1000 * height,
                                   x2 / 1000 * width, y2 / 1000 * height,
                                   current))
    return [(k, dedupe(groups[k]) if merge else groups[k]) for k in order]


class LocateAnything:
    """Loads the model once. Import torch lazily so this file is safe to import."""

    def __init__(self, model_path: str = MODEL_ID, device: str = "cuda", dtype=None):
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        self.torch = torch
        self.dtype = dtype or torch.bfloat16
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=self.dtype, trust_remote_code=True,
        ).to(device).eval()

    def predict(self, image, question: str, generation_mode: str = "hybrid",
                max_new_tokens: int = 384) -> str:
        torch = self.torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            resp = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                do_sample=False,              # deterministic: this feeds a robot
                verbose=False,
            )
        return resp[0] if isinstance(resp, tuple) else resp

    # ------------------------------------------------------------------ tasks
    def locate(self, image, phrase: str, **kw) -> list[Box]:
        """All instances matching a phrase, as pixel boxes."""
        answer = self.predict(
            image, "Locate all the instances that match the following description: %s." % phrase, **kw)
        return parse_boxes(answer, image.width, image.height, phrase)

    def locate_one(self, image, phrase: str, **kw) -> Box | None:
        """The single best instance -- what face tracking wants."""
        answer = self.predict(
            image, "Locate a single instance that matches the following description: %s." % phrase, **kw)
        boxes = parse_boxes(answer, image.width, image.height, phrase)
        return max(boxes, key=lambda b: b.area) if boxes else None

    def detect(self, image, categories: list[str], **kw) -> list[Box]:
        """Multi-category detection, flattened. Prefer detect_labelled()."""
        return [b for _, bs in self.detect_labelled(image, categories, **kw) for b in bs]

    def detect_labelled(self, image, categories: list[str],
                        **kw) -> list[tuple[str, list[Box]]]:
        """Multi-category detection, keeping which box belongs to which category.

        Categories are joined with the model's own "</c>". A comma-joined string
        is read as one long phrase instead and the model falls back to emitting
        regions -- measured 9 boxes, 6 of them junk, against 4 correct ones.

        Category ORDER matters, and not for the reason you would guess. The
        model can get stuck repeating one box: asked for
        "food pieces</c>bowl</c>fork" it emitted the bowl ~60 times, exhausted
        max_new_tokens, and never reached "fork" at all. Dedupe hides the
        repeats but cannot recover the category that was starved. So put what
        you actually need FIRST, and give it room.
        """
        answer = self.predict(
            image, "Locate all the instances that matches the following description: %s."
            % "</c>".join(categories), **kw)
        return parse_labelled(answer, image.width, image.height)

    # -------------------------------------------------------------- measuring
    def benchmark(self, image, phrase: str = "face", n: int = 5,
                  generation_mode: str = "hybrid") -> dict:
        """Latency over n runs, first run reported separately.

        The first call pays warm-up and is not representative; quoting it as the
        steady-state rate would overstate the cost several-fold.
        """
        times = []
        for i in range(n + 1):
            t0 = time.perf_counter()
            boxes = self.locate_one(image, phrase, generation_mode=generation_mode)
            self.torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        warm, steady = times[0], sorted(times[1:])
        med = steady[len(steady) // 2]
        return dict(warmup_s=warm, median_s=med, min_s=steady[0], max_s=steady[-1],
                    hz=1.0 / med if med else float("inf"), found=boxes is not None,
                    mode=generation_mode)
