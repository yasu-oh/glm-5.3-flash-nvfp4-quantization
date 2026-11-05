# GLM-5.3-Flash NVFP4 Quantization

Quantizes `zai-org/GLM-5.3-Flash-BF16` MoE experts using `GPTQModifier`, `nvfp4_expanded_imatrix` (`strict=True`), and FP8 KV-cache quantization.

Calibration uses 1,024 chat-template-normalized samples: 512 from `CohereLabs/aya_dataset` and 512 from `interstellarninja/hermes_reasoning_tool_use`.

## Usage

Requires a CUDA GPU and current packages with GLM5Next support.

```bash
pip install --upgrade datasets pyyaml
pip install --upgrade --pre compressed-tensors llmcompressor
cp sample_config.yaml config.yaml
# Edit model_id, device_map, save_dir, and use_bundled_chat_template.
python glm-5.3-flash-nvfp4-quantization.py
```

`use_bundled_chat_template: true` uses the bundled GLM template; `false` uses the model default.

## Tests

```bash
pip install transformers datasets pyyaml
python -m unittest -v
```

Tests run on CPU using tokenizer files and real datasets, without downloading model weights. GPU quantization and inference compatibility are not verified by these tests.
