"""CPU orchestration tests with mocked GPU libraries, not distributed GPU tests."""

import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call, patch

SCRIPT = Path(__file__).with_name("glm-5.3-flash-nvfp4-quantization.py")
spec = importlib.util.spec_from_file_location("distributed_quantization", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(SCRIPT)
QUANTIZATION = importlib.util.module_from_spec(spec)
spec.loader.exec_module(QUANTIZATION)


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        self.torch = Mock()
        self.torch.cuda.is_available.return_value = True
        self.torch.distributed.is_initialized.return_value = True
        self.init_dist = Mock()
        self.tokenizer_class = Mock()
        self.model_class = Mock()
        self.oneshot = Mock()
        self.dataset = ["sample"]
        self.load_context = Mock(return_value=nullcontext())
        self.config = ("model", "auto_offload", Path("output"), Path("offload"), False)
        self.calls.attach_mock(self.init_dist, "init_dist")
        self.calls.attach_mock(self.tokenizer_class.from_pretrained, "load_tokenizer")
        self.calls.attach_mock(self.load_context, "load_context")
        modules = {}
        for name, attributes in {
            "torch": {"cuda": self.torch.cuda, "distributed": self.torch.distributed},
            "compressed_tensors.offload": {"init_dist": self.init_dist},
            "compressed_tensors.quantization": {
                "QuantizationArgs": Mock(),
                "preset_name_to_scheme": Mock(),
            },
            "llmcompressor": {"oneshot": self.oneshot},
            "llmcompressor.modifiers.gptq": {"GPTQModifier": Mock()},
            "llmcompressor.utils": {"load_context": self.load_context},
            "transformers": {
                "AutoTokenizer": self.tokenizer_class,
                "Glm5NextForConditionalGeneration": self.model_class,
            },
        }.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(QUANTIZATION, "load_config", return_value=self.config))
        self.enterContext(patch.object(QUANTIZATION, "apply_chat_template_file"))
        self.enterContext(patch.object(QUANTIZATION, "create_calibration_dataset", return_value=self.dataset))

    def test_all_ranks_save_model_but_only_rank_zero_saves_tokenizer(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self.calls.reset_mock()
                self.torch.distributed.reset_mock()
                self.torch.distributed.get_rank.return_value = rank
                self.model_class.reset_mock()
                self.tokenizer_class.reset_mock()
                self.oneshot.reset_mock()
                QUANTIZATION.main()
                self.init_dist.assert_called_once_with()
                self.assertEqual(self.calls.mock_calls[0], call.init_dist())
                self.load_context.assert_called_once_with(self.model_class)
                self.model_class.from_pretrained.assert_called_once_with(
                    "model", dtype="auto", device_map="auto_offload", offload_folder="offload"
                )
                # Model saving performs collectives internally: never guard it by rank.
                self.model_class.from_pretrained.return_value.save_pretrained.assert_called_once_with(
                    Path("output"), save_compressed=True
                )
                tokenizer_save = self.tokenizer_class.from_pretrained.return_value.save_pretrained
                self.assertEqual(tokenizer_save.call_count, 1 if rank == 0 else 0)
                if rank == 0:
                    tokenizer_save.assert_called_once_with(Path("output"))
                kwargs = self.oneshot.call_args.kwargs
                self.assertIs(kwargs["dataset"], self.dataset)
                self.assertEqual(kwargs["num_calibration_samples"], len(self.dataset))
                self.assertEqual(kwargs["max_seq_length"], 4096)
                self.torch.distributed.destroy_process_group.assert_called_once_with()

    def test_cleanup_and_original_exception_on_failure(self):
        for operation in (
            self.init_dist,
            self.tokenizer_class.from_pretrained,
            self.model_class.from_pretrained,
            self.oneshot,
            self.model_class.from_pretrained.return_value.save_pretrained,
        ):
            with self.subTest(operation=operation):
                self.torch.distributed.destroy_process_group.reset_mock()
                operation.side_effect = RuntimeError("expected failure")
                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    QUANTIZATION.main()
                self.torch.distributed.destroy_process_group.assert_called_once_with()
                operation.side_effect = None

    def test_failed_initialization_without_process_group_needs_no_cleanup(self):
        self.torch.distributed.is_initialized.return_value = False
        self.init_dist.side_effect = ValueError("torchrun required")
        with self.assertRaisesRegex(ValueError, "torchrun required"):
            QUANTIZATION.main()
        self.torch.distributed.destroy_process_group.assert_not_called()
        self.model_class.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
