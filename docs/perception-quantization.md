# Quantizing LocateAnything on the Jetson — measured

All numbers from the AGX Orin (`jetson`), `torch 2.14.0+cu130`, `torchao 0.18.0`,
2026-09-22. Microbenchmarks are a single `nn.Linear` at decode shape; treat them
as an upper bound on the end-to-end win, not a prediction of it.

## The hardware decides more than the preference does

| | Orin AGX (sm_87) | triphammer RTX 4070 Ti (sm_89) |
|---|---|---|
| FP8 | **unavailable** | available |
| INT8 | **native** | native |
| INT4 | via tinygemm only | available |

Measured, not read off a datasheet:

```
FP8 torch._scaled_mm : UNAVAILABLE -> requires compute capability >= 9.0 or 8.9
INT8 torch._int_mm   : WORKS (256,256) int32
```

FP8 tensor cores start at Ada/Hopper. Orin is Ampere-class, compute capability
8.7, so FP8 is not an option on the box that is going to run this — even though
it is an option on the GPU box that is not.

## torch.compile is not optional

The single most important result here. Same config, same shape, batch 1,
2048 → 11008 (a real FFN projection of a 3B model):

| | latency | vs bf16 | rel err |
|---|---|---|---|
| bf16 eager | 0.271 ms | 1.00× | — |
| bf16 compiled | 0.275 ms | 0.99× | — |
| int8 weight-only **eager** | 1.255 ms | **0.22×** | 0.0049 |
| int8 weight-only **+compile** | **0.155 ms** | **1.75×** | 0.0046 |

In eager mode quantization is a **4.5× pessimization**: the dequantise step
never fuses, so every matmul pays an extra full-width pass over the weights.
Compiled, inductor fuses it and the memory-bandwidth saving finally shows up.

Corollary: any benchmark of quantized inference on this stack that does not use
`torch.compile` is measuring the wrong thing. An early version of this
comparison did exactly that and concluded INT8 was slower than BF16.

## Shape matters too

| shape (batch 1) | bf16 | int8+compile |
|---|---|---|
| 2048 → 11008 | 0.271 ms | **0.155 ms (1.75×)** |
| 2048 → 2048 | 0.077 ms | 0.136 ms (0.57×) |

Quantization pays off on the wide projections that dominate a 3B model and
loses on small square ones, where the fixed overhead is not amortised. A
whole-model `quantize_` will therefore not deliver 1.75× end to end, and
filtering which layers get quantized is worth trying.

## INT8 beats INT4 here, on both axes

| variant (batch 1, 2048 → 11008, compiled) | latency | vs bf16 | rel err |
|---|---|---|---|
| **int8 weight-only** | **0.155 ms** | **1.75×** | **0.0046** |
| int4 `tile_packed_to_4d` | 0.191 ms | 1.40× | 0.0652 |
| int4 `plain` / `preshuffled` | — | — | needs `mslk` (no aarch64 build) |
| int4 `plain_int32` | — | — | no CUDA support in torchao 0.18 |

INT4 runs, through the tinygemm `tile_packed_to_4d` path, and is **slower than
INT8 and 14× less accurate**. The int8 path lands on `_int_mm`, which Ampere
implements in hardware; the int4 path goes through PTX JIT and pays group-wise
dequant overhead that the narrower weights do not repay at this size.

That 0.065 relative error is the number to watch. This model is a *detector*:
weight error becomes box displacement, and a box centre that moves is a fork
that moves. The end-to-end test must report **box centre shift against the BF16
reference**, not just latency — a variant that is faster and puts the mouth 15
px from where it is has failed.

## The foundation is officially unsupported

```
arch list: ['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']
  8.0 supports hardware CC >=8.0,<9.0  except {8.7}
No published PyTorch CUDA builds for 2.14.0+cu130 support this GPU
device: Orin (8, 7)   -> kernels run anyway, via PTX JIT from sm_80
```

NVIDIA's Jetson wheel index (`pypi.jetson-ai-lab.io`) carries `jp6/cu126|cu128|
cu129` only — nothing for JetPack 7 / CUDA 13.2, which is what this board runs.
The upstream `manylinux_2_28_aarch64` wheel works because the driver JIT-compiles
sm_80 PTX, and that is exactly why some hand-written kernels (the `mslk` int4
paths) are missing while generic ones are fine.

If this becomes limiting, the options are a JetPack-native build when NVIDIA
publishes one, or building torch from source for `sm_87`.

## Also worth knowing

- `TensorRT`, `TensorRT-LLM` and `Triton` are **not supported** by
  LocateAnything (NVIDIA's own model card). The TRT-LLM INT4 route does not
  exist for this model, which is why quantization goes through torchao.
- `MagiAttention` is Hopper/Blackwell only; on Orin the MTP path falls back to
  PyTorch SDPA — functional, slower.
- Architecture §7 specified INT4 because triphammer has 12 GB. On a 61 GB
  unified-memory Orin, memory is not the constraint; **latency is**, and that is
  a different argument reaching a different conclusion (INT8, compiled).
