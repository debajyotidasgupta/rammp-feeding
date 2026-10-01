"""Open-vocabulary detection with nvidia/LocateAnything-3B.

Two jobs in the feeding stack, with very different duty cycles:

  food   -- once per bite cycle, ~every 15 s. Latency barely matters.
  face   -- continuously during TRANSPORT/PRESENT. Latency is the whole game.

The face is MediaPipe's (features.face_detector), with this model as its fallback: it does not reach the face
path's rate on this hardware. Perception stays out of the control loop, so a slow detector delays the next goal
rather than stalling a trajectory.

Model facts:

  * Coordinates come back as normalised integers in [0, 1000] inside
    `<box><x1><y1><x2><y2></box>`, NOT pixels. Scale by the image size.
  * `generation_mode="hybrid"` is the default NVIDIA recommends: parallel box
    decoding with an autoregressive fallback when the format comes out
    irregular. "fast" is parallel-only and can truncate.
  * MagiAttention is Hopper/Blackwell only. On Orin we fall back to PyTorch
    SDPA -- functional, slower for the multi-token-prediction path.
  * TensorRT / TensorRT-LLM / Triton are NOT supported by this model, so it
    runs as BF16 (7.7 GB of weights; fine on the 61 GB unified-memory Orin).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MODEL_ID = "nvidia/LocateAnything-3B"
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# The model tags each category before its boxes:
#   <ref>food pieces</ref><box>..</box><box>..</box><ref>bowl</ref><box>..</box>
# walk both tokens in order so each box keeps its category.
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
    """Greedy NMS: the model emits the same object many times over."""
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

    # ------------------------------------------------------------------ stop on repeats
    def _stop_on_repeat(self, iou_thresh: float = 0.3, width: int = 1280, height: int = 720):
        """While active, end generation at the first box that repeats an earlier one.

        Past the real instances the model keeps emitting near-copies of one box until max_new_tokens. Its
        loop has no stopping hook, but every multi-token step goes through the module-level handle_pattern();
        returning its own 'im_end' there ends generation exactly as the model would. Slivers (< 20 px or
        aspect > 3) also stop it. Restored on exit."""
        import contextlib
        import sys as _sys
        mod = _sys.modules[type(self.model).__module__]
        orig = mod.handle_pattern
        tid = self.model.token_ids
        c0, c1 = tid["coord_start_token_id"], tid["coord_end_token_id"]
        seen: list[tuple] = []

        def iou(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
            return ix * iy / u if u > 0 else 0.0

        def patched(x0, token_ids, generation_mode="hybrid"):
            out = orig(x0, token_ids, generation_mode)
            toks = out.get("tokens", [])
            coords = [t - c0 for t in toks if c0 <= t <= c1]
            if len(coords) == 4:
                box = tuple(coords)
                bw, bh = (box[2] - box[0]) * width / 1000.0, (box[3] - box[1]) * height / 1000.0
                sliver = min(bw, bh) < 20 or max(bw, bh) > 3 * max(1e-6, min(bw, bh))
                if sliver or any(iou(box, b) > iou_thresh for b in seen):
                    return {"type": "im_end", "tokens": [tid["im_end_token_id"]], "need_switch_to_ar": False,
                            "is_terminal": True}
                seen.append(box)
            return out

        @contextlib.contextmanager
        def ctx():
            mod.handle_pattern = patched
            try:
                yield
            finally:
                mod.handle_pattern = orig
        return ctx()

    # ------------------------------------------------------------------ tasks
    def locate(self, image, phrase: str, stop_on_repeat: bool = True, **kw) -> list[Box]:
        """All instances matching a phrase, as pixel boxes."""
        q = "Locate all the instances that match the following description: %s." % phrase
        if stop_on_repeat:
            with self._stop_on_repeat(width=image.width, height=image.height):
                answer = self.predict(image, q, **kw)
        else:
            answer = self.predict(image, q, **kw)
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

        Categories are joined with the model's own "</c>" (a comma-joined string
        reads as one long phrase). Order matters: the model can repeat one box
        until max_new_tokens and never reach a later category, so put what you
        need FIRST and give it room.
        """
        answer = self.predict(
            image, "Locate all the instances that matches the following description: %s."
            % "</c>".join(categories), **kw)
        return parse_labelled(answer, image.width, image.height)

