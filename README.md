# GLM-5.3-Flash NVFP4 Quantization

Quantizes `zai-org/GLM-5.3-Flash-BF16` MoE experts using `GPTQModifier`, `nvfp4_expanded_imatrix` (`strict=True`).

Calibration: 512 samples each from `CohereLabs/aya_dataset` and `interstellarninja/hermes_reasoning_tool_use`.

## Usage

Requires CUDA and current packages with GLM5Next support.

```bash
pip install 'flash-linear-attention[cuda]' datasets pyyaml transformers
pip install causal-conv1d --no-build-isolation
pip install --pre compressed-tensors llmcompressor
cp sample_config.yaml config.yaml
# Edit config.yaml. Set nproc-per-node to the number of GPUs (1 for a single GPU).
torchrun --standalone --nproc-per-node=2 glm-5.3-flash-nvfp4-quantization.py
```

Use `device_map: auto_offload` for CPU offloading. `use_bundled_chat_template: true` uses the bundled template; `false` uses the model default.

## Tests

```bash
pip install transformers datasets pyyaml
python -m unittest -v
```

CPU tests only; GPU quantization and inference compatibility are not verified.
