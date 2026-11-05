import ast
import importlib.util
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import calibration_dataset as DATASET

ROOT = Path(__file__).parent
SCRIPT = ROOT / "glm-5.3-flash-nvfp4-quantization.py"
MODEL_ID = "zai-org/GLM-5.3-Flash-BF16"
spec = importlib.util.spec_from_file_location("quantization", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(SCRIPT)
QUANTIZATION = importlib.util.module_from_spec(spec)
spec.loader.exec_module(QUANTIZATION)


def call(name, arguments=None, **extra):
    return {
        "from": "gpt",
        "value": "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments or {}, **extra})
        + "</tool_call>",
    }


def response(content, **extra):
    return {
        "from": "tool",
        "value": "<tool_response>"
        + json.dumps({"content": content, **extra})
        + "</tool_response>",
    }


class NormalizationTests(unittest.TestCase):
    def test_config_and_template_selection(self):
        class Tokenizer:
            chat_template = "default"

        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            device_map = {"model.embed_tokens": "cpu", "model.layers.0": 0}
            config = {
                "model_id": MODEL_ID,
                "device_map": device_map,
                "save_dir": "output",
                "use_bundled_chat_template": False,
            }
            (work_dir / "config.yaml").write_text(json.dumps(config))
            self.assertEqual(
                QUANTIZATION.load_config(work_dir),
                (MODEL_ID, device_map, work_dir / "output", False),
            )
            config["use_bundled_chat_template"] = "true"
            (work_dir / "config.yaml").write_text(json.dumps(config))
            with self.assertRaises(TypeError):
                QUANTIZATION.load_config(work_dir)
            tokenizer = Tokenizer()
            QUANTIZATION.apply_chat_template_file(tokenizer, work_dir, False)
            self.assertEqual(tokenizer.chat_template, "default")
            with self.assertRaises(FileNotFoundError):
                QUANTIZATION.apply_chat_template_file(tokenizer, work_dir, True)
            (work_dir / "chat_template.jinja").write_text("bundled")
            QUANTIZATION.apply_chat_template_file(tokenizer, work_dir, True)
            self.assertEqual(tokenizer.chat_template, "bundled")

    def test_tools_and_thinking(self):
        tools = DATASET.normalize_tools(
            '[{"name":"search","parameters":{"count":{"type":"int"},'
            '"queries":{"type":"List[str]"}}}]'
        )
        self.assertEqual(
            tools[0]["function"]["parameters"]["properties"],
            {
                "count": {"type": "integer"},
                "queries": {"type": "array", "items": {"type": "string"}},
            },
        )
        messages = DATASET.normalize_conversation(
            [
                {
                    "from": "system",
                    "value": "Keep this.\nYou are a function calling AI model. <tools>[]</tools>",
                },
                {"from": "human", "value": "Check Tokyo"},
                {
                    "from": "gpt",
                    "value": '<think>Mention `<tool_call>` only as text.</think><tool_call>{"name":"weather","arguments":"{\\"city\\":\\"Tokyo\\"}"}</tool_call>',
                },
                response({"temperature": 28}, name="weather", tool_call_id="legacy-id"),
            ],
            strict_tool_data=True,
        )
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "tool"]
        )
        self.assertEqual(messages[0]["content"], "Keep this.")
        self.assertEqual(
            messages[2]["reasoning_content"], "Mention `<tool_call>` only as text."
        )
        self.assertEqual(
            messages[2]["tool_calls"][0]["function"]["arguments"], {"city": "Tokyo"}
        )
        self.assertEqual(json.loads(messages[3]["content"]), {"temperature": 28})
        self.assertEqual(
            messages[3]["tool_call_id"], messages[2]["tool_calls"][0]["id"]
        )

    def test_parallel_results_match_ids_and_preserve_identical_outputs(self):
        first = call("weather", {"city": "Tokyo"}, id="tokyo")
        second = call("weather", {"city": "Osaka"}, id="osaka")
        first["value"] += second["value"]
        messages = DATASET.normalize_conversation(
            [
                first,
                response({"temperature": 29}, name="weather", tool_call_id="osaka"),
                response({"temperature": 29}, name="weather", tool_call_id="tokyo"),
                response({"temperature": 29}, name="weather", tool_call_id="tokyo"),
            ],
            strict_tool_data=True,
        )
        self.assertEqual([m["role"] for m in messages], ["assistant", "tool", "tool"])
        self.assertEqual([m["tool_call_id"] for m in messages[1:]], ["osaka", "tokyo"])
        self.assertEqual(
            [c["id"] for c in messages[0]["tool_calls"]], ["tokyo", "osaka"]
        )

    def test_repeated_calls_do_not_lose_identical_results(self):
        messages = DATASET.normalize_conversation(
            [
                call("weather"),
                response("sunny", name="weather"),
                {"from": "gpt", "value": "Sunny today."},
                {"from": "human", "value": "Check again"},
                call("weather"),
                response("sunny", name="weather"),
            ],
            strict_tool_data=True,
        )
        self.assertEqual(
            [m["content"] for m in messages if m["role"] == "tool"], ["sunny", "sunny"]
        )

    def test_embedded_responses_and_unwrapped_payload(self):
        for payload in (None, False, 0, {"name": "Bistro", "distance": 500}):
            with self.subTest(payload=payload):
                messages = DATASET.normalize_conversation(
                    [
                        call("lookup"),
                        {
                            "from": "gpt",
                            "value": "<think>Review result.</think><tool_response>"
                            + json.dumps({"name": "lookup", "content": payload})
                            + "</tool_response>Final answer.",
                        },
                    ],
                    strict_tool_data=True,
                )
                self.assertEqual(
                    [m["role"] for m in messages], ["assistant", "tool", "assistant"]
                )
                self.assertEqual(json.loads(messages[1]["content"]), payload)
                self.assertEqual(messages[2]["content"], "Final answer.")
                self.assertEqual(messages[2]["reasoning_content"], "Review result.")
        messages = DATASET.normalize_conversation(
            [
                call("restaurant"),
                {
                    "from": "tool",
                    "value": "<tool_response>{'name':'Bistro','distance':500}</tool_response>",
                },
                {
                    "from": "gpt",
                    "value": '<tool_response>**{"name":"restaurant","content":{"name":"Bistro","distance":500}}**</tool_response>',
                },
            ],
            strict_tool_data=True,
        )
        self.assertEqual([m["role"] for m in messages], ["assistant", "tool"])
        self.assertEqual(
            json.loads(messages[1]["content"]), {"name": "Bistro", "distance": 500}
        )

    def test_invalid_tool_data_is_filtered(self):
        conversations = [
            [
                call("expected", id="shared"),
                response(True, name="wrong", tool_call_id="shared"),
            ],
            [
                {
                    "from": "gpt",
                    "value": call("first")["value"] + call("second")["value"],
                },
                response(True),
            ],
            [
                call("overview"),
                {
                    "from": "tool",
                    "value": '<tool_response>{"name":"overview","content":"{"broken"}"}</tool_response>',
                },
            ],
            [
                {
                    "from": "gpt",
                    "value": '<tool_call>{"name":"broken","arguments":[}</tool_call>',
                }
            ],
            [{"from": "gpt", "value": '<tool_call>{"arguments":{}}</tool_call>'}],
            [
                call("old"),
                {"from": "human", "value": "New question"},
                response(True, name="old"),
            ],
            [
                call("lookup"),
                {
                    "from": "tool",
                    "value": "<tool_response>{'name':'lookup','content':{'items':[...]}}</tool_response>",
                },
            ],
        ]
        for conversation in conversations:
            with self.subTest(conversation=conversation):
                self.assertFalse(DATASET.has_consistent_tool_data(conversation))
                with self.assertRaises(DATASET.InvalidToolDataError):
                    DATASET.normalize_conversation(conversation, strict_tool_data=True)

    def test_quantization_scope(self):
        # Static recipe contract only: this does not execute GPU quantization.
        tree = ast.parse(SCRIPT.read_text())
        distributed_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "init_dist"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get_rank", "destroy_process_group"}
            )
        ]
        self.assertEqual(
            distributed_calls,
            [],
            "Plain python execution must not require a process group",
        )
        modifier = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GPTQModifier"
        )
        kwargs = {kw.arg: kw.value for kw in modifier.keywords}
        preset = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "preset_name_to_scheme"
        )
        self.assertEqual(ast.literal_eval(preset.args[0]), "NVFP4")
        targets = ast.literal_eval(preset.args[1])
        attributes = {
            ast.unparse(node.targets[0]): node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
        }
        self.assertEqual(
            ast.literal_eval(attributes["nvfp4_scheme.weights.observer"]),
            "nvfp4_expanded_imatrix",
        )
        self.assertEqual(
            ast.literal_eval(attributes["nvfp4_scheme.weights.observer_kwargs"]),
            {"strict": True},
        )
        self.assertEqual(
            ast.unparse(kwargs["config_groups"]), "{'group_0': nvfp4_scheme}"
        )
        for name in (
            "model.language_model.layers.3.mlp.experts.0.gate_proj",
            "model.language_model.layers.3.mlp.experts.1.up_proj",
            "model.language_model.layers.3.mlp.experts.2.down_proj",
        ):
            self.assertTrue(
                any(re.match(pattern.removeprefix("re:"), name) for pattern in targets)
            )
        for name in (
            "lm_head",
            "model.visual.blocks.0.mlp.up_proj",
            "model.language_model.layers.3.mlp.gate",
            "model.language_model.layers.3.self_attn.indexer.wq_b",
            "model.language_model.layers.0.mlp.up_proj",
        ):
            self.assertFalse(
                any(re.match(pattern.removeprefix("re:"), name) for pattern in targets)
            )
        self.assertEqual(
            ast.literal_eval(kwargs["ignore"]),
            [
                r"re:.*visual.*",
                "lm_head",
                r"re:.*mlp\.gate$",
                r"re:.*self_attn\.indexer\..*",
            ],
        )


class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicit allowlist: never fetch model weights or remote Python code.
        path = snapshot_download(
            MODEL_ID,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ],
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        QUANTIZATION.apply_chat_template_file(cls.tokenizer, ROOT, True)

    def test_glm_rendering_contract(self):
        tokenizer = self.tokenizer
        conversation = [
            {"from": "human", "value": "Check weather"},
            {
                "from": "gpt",
                "value": "<think>Check both.</think>"
                + call(
                    "weather",
                    {"city": "東京", "days": [1, 2], "rain": False},
                    id="tokyo",
                )["value"]
                + call("weather", {"city": "Osaka"}, id="osaka")["value"],
            },
            response({"city": "Osaka"}, name="weather", tool_call_id="osaka"),
            response({"city": "東京"}, name="weather", tool_call_id="tokyo"),
            {"from": "gpt", "value": "Done."},
            {"from": "human", "value": "Thanks"},
            {"from": "gpt", "value": "Welcome."},
        ]
        messages = DATASET.normalize_conversation(conversation, strict_tool_data=True)
        tools = DATASET.normalize_tools([{"name": "weather", "parameters": {}}])
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            clear_thinking=False,
        )
        self.assertTrue(
            rendered.startswith("[gMASK]<sop><|system|>Reasoning Effort: Max")
        )
        self.assertIn("<think>Check both.</think>", rendered)
        self.assertIn("<|assistant|><think></think>Done.", rendered)
        self.assertIn("<|assistant|><think></think>Welcome.", rendered)
        self.assertIn(
            "<tool_call>weather<arg_key>city</arg_key><arg_value>東京</arg_value><arg_key>days</arg_key><arg_value>[1, 2]</arg_value><arg_key>rain</arg_key><arg_value>false</arg_value></tool_call>",
            rendered,
        )
        self.assertIn(
            '<|observation|><tool_response>{"city": "東京"}</tool_response><tool_response>{"city": "Osaka"}</tool_response>',
            rendered,
        )
        self.assertFalse(rendered.endswith("<|assistant|><think>"))
        self.assertEqual(
            tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
                clear_thinking=False,
                return_dict=False,
            ),
            tokenizer.encode(rendered, add_special_tokens=False),
        )

    def test_embedded_response_before_next_call(self):
        for response_id in ("tokyo", "legacy-id", None):
            with self.subTest(response_id=response_id):
                first = call("weather", {"city": "Tokyo"}, id="tokyo")
                second = call("weather", {"city": "Osaka"}, id="osaka")
                second["value"] = (
                    response(
                        {"city": "Tokyo"}, name="weather", tool_call_id=response_id
                    )["value"]
                    + second["value"]
                )
                messages = DATASET.normalize_conversation(
                    [
                        first,
                        second,
                        response(
                            {"city": "Osaka"}, name="weather", tool_call_id="osaka"
                        ),
                    ],
                    strict_tool_data=True,
                )
                self.assertEqual(
                    [m["role"] for m in messages],
                    ["assistant", "tool", "assistant", "tool"],
                )
                self.assertEqual(
                    [m["tool_call_id"] for m in messages if m["role"] == "tool"],
                    ["tokyo", "osaka"],
                )
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                self.assertIn(
                    '<|observation|><tool_response>{"city": "Tokyo"}</tool_response><|assistant|><think></think><tool_call>weather<arg_key>city</arg_key><arg_value>Osaka</arg_value></tool_call>',
                    rendered,
                )
        # A result following a call in the same source turn belongs to that call.
        current = call("weather", id="current")
        current["value"] += response("sunny", name="weather", tool_call_id="current")[
            "value"
        ]
        messages = DATASET.normalize_conversation([current], strict_tool_data=True)
        self.assertEqual([m["role"] for m in messages], ["assistant", "tool"])
        self.assertEqual(messages[1]["tool_call_id"], "current")

    def test_real_calibration_samples(self):
        tokenizer = self.tokenizer
        selected = DATASET.select_hermes_samples(
            load_dataset(DATASET.HERMES_DATASET, split="train")
        )
        self.assertEqual(
            Counter(selected["scenario_category"]),
            {
                "single": 102,
                "multiturn": 154,
                "multistep": 205,
                "relevance": 51,
            },
        )
        response_count = 0
        for index, row in enumerate(selected):
            with self.subTest(index=index, category=row["scenario_category"]):
                messages = DATASET.normalize_conversation(
                    row["conversations"], strict_tool_data=True
                )
                results = [m for m in messages if m["role"] == "tool"]
                actual_payloads = [m["content"] for m in results]
                for turn in row["conversations"]:
                    visible = DATASET.THINK.sub("", turn["value"])
                    raw_results = DATASET.TOOL_RESPONSE.findall(visible)
                    if turn["from"] == "tool" and not raw_results:
                        raw_results = [visible]
                    for raw in raw_results:
                        parsed = DATASET.parse_response(raw)
                        if (
                            turn["from"] == "tool"
                            or parsed.get("name")
                            or parsed.get("tool_call_id")
                        ):
                            payload = parsed.get("content")
                            expected = (
                                payload
                                if isinstance(payload, str)
                                else json.dumps(payload, ensure_ascii=False)
                            )
                            self.assertIn(expected, actual_payloads)
                for m in messages:
                    self.assertNotIn("tool_responses", m)
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=DATASET.normalize_tools(row["tools"]),
                    tokenize=False,
                    add_generation_prompt=False,
                    clear_thinking=False,
                )
                # Tag mentions in user/reasoning text are not actual tool results.
                for result in results:
                    self.assertIn(
                        "<tool_response>" + result["content"] + "</tool_response>",
                        rendered,
                    )
                self.assertLessEqual(
                    len(tokenizer.encode(rendered, add_special_tokens=False)), 4096
                )
                response_count += len(results)
        self.assertGreater(response_count, 0)
        calibration = DATASET.create_calibration_dataset(tokenizer)
        self.assertEqual(len(calibration), 1024)
        self.assertEqual(calibration.column_names, ["text"])
        lengths = [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in calibration["text"]
        ]
        self.assertTrue(all(lengths))
        print(
            f"\nCalibration: {len(calibration)} samples, {response_count} tool responses; "
            f"token lengths {min(lengths)}..{max(lengths)}; "
            f"{sum(length > 4096 for length in lengths)} samples above oneshot's 4096-token limit"
        )


if __name__ == "__main__":
    unittest.main()
