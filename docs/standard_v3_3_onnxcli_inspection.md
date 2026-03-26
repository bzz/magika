# `standard_v3_3` ONNX architecture inspection (using `onnxcli`)

## What I ran

```bash
python -m pip install onnxcli
onnxcli inspect -m -io -n assets/models/standard_v3_3/model.onnx
```

The `onnxcli` metadata + I/O summary for this model reports:

- input tensor: `bytes` with shape `[batch, 2048]` (`INT32`)
- output tensor: `target_label` with shape `[batch, 214]` (`FLOAT`)
- total nodes: 95
- ONNX opset: 15
- producer: `tf2onnx`

## Architecture recovered from ONNX graph

From `onnxcli inspect` node names and the ONNX initializer tensor shapes:

1. **Byte-token embedding stage**
   - `MagikaV2/Dense_0/einsum/Einsum` (`MatMul`) with weights shaped `[257, 64]`.
   - This indicates byte vocabulary + padding token (`0..256`) projected to a 64-d embedding.
2. **Activation + normalization**
   - GELU-like activation subgraph (`Mul/Add/Tanh` pattern).
   - `LayerNorm_0` block appears immediately after embedding.
3. **Sequence mixing stage**
   - `MagikaV2/Conv_0/Conv2D` with kernel tensor shape `[512, 256, 5, 1]`.
   - Followed by a second GELU-like block (`ApplyActivation_1`) and `GlobalMaxPool`.
4. **Head**
   - `LayerNorm_1`.
   - Final classifier `MagikaV2/Dense_1/MatMul` with weights `[512, 214]` + bias `[1, 214]`.
   - Final softmax computation is represented explicitly as `ReduceMax` -> `Exp` -> `ReduceSum` -> `Div`.

## How this maps to the ICSE 2025 paper

The paper's Figure 3 / Section IV-B describes the core Magika architecture as:

- **Input construction:** three windows of 512 bytes (beginning/middle/end), concatenated to `3x512=1536` tokens, one-hot encoded.
- **Embedding/trunk:** dense embedding, reshape to `384x512`, dense layers + global max pooling, then final softmax classifier.
- **Output layer:** one probability per content type.

So the inspected `standard_v3_3` ONNX graph is **structurally consistent** with the paper's high-level recipe (token embedding -> trunk processing -> pooling -> dense classification -> softmax), but with version-specific differences:

- Current model input is `2048` tokens, not `1536`.
- Current model outputs `214` classes.
- Current graph includes an explicit convolutional mixing block (`Conv2D`) in the trunk representation.

Those differences align with this repo's shipped model config (`beg_size=1024`, `mid_size=0`, `end_size=1024`) and updated label space for `standard_v3_3`.
