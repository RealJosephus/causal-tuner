"""Serialized fast-tokenizer assets and sandboxed training chat templates."""

import hashlib
import json
import math
import re
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path
from tokenizers import Tokenizer
from jinja2 import nodes, TemplateError
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

MARKER = "\ue000CTGEN:"
TOKENIZER_PATTERNS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                      "chat_template.json", "chat_template.jinja", "chat_templates/*.jinja")


def tokenizer_files(directory):
    return sorted({path for pattern in TOKENIZER_PATTERNS for path in Path(directory).glob(pattern) if path.is_file()})


class Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        line = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("capture"), [], [], body).set_lineno(line)

    def capture(self, caller):
        return MARKER+"START\ue001"+caller()+MARKER+"END\ue001"


@lru_cache(maxsize=32)
def compiled_template(source):
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[Generation, "jinja2.ext.loopcontrols"])
    def fail(message):
        raise TemplateError(message)
    environment.globals.update(raise_exception=fail,
                               strftime_now=lambda format: fail("Dynamic wall-clock templates are not replayable; supply a fixed date through template variables"))
    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)
    environment.filters["tojson"] = tojson
    return environment.from_string(source)


class TrainingTokenizer:
    def __init__(self, directory):
        directory = Path(directory)
        self.backend = Tokenizer.from_file(str(directory/"tokenizer.json"))
        self.backend.no_padding()
        self.backend.no_truncation()
        def read(name):
            path = directory/name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        config = read("tokenizer_config.json")
        special = {**read("special_tokens_map.json"), **config}
        self.special = {k: (v["content"] if isinstance(v, dict) else v) for k, v in special.items()
                        if k.endswith("_token") and (isinstance(v, str) or isinstance(v, dict))}
        self.template = config.get("chat_template") or read("chat_template.json") or None
        files = {p.stem: p.read_text(encoding="utf-8") for p in sorted((directory/"chat_templates").glob("*.jinja"))}
        if (directory/"chat_template.jinja").exists():
            files["default"] = (directory/"chat_template.jinja").read_text(encoding="utf-8")
        if files:
            self.template = files
        if isinstance(self.template, list):
            self.template = {item["name"]: item["template"] for item in self.template}
        self.eos_token_id = self.backend.token_to_id(self.special.get("eos_token", ""))
        self.pad_token_id = self.backend.token_to_id(self.special.get("pad_token", ""))
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer assets must define a pad or EOS token")
        digest = hashlib.sha256()
        for path in tokenizer_files(directory):
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(path.read_bytes())
        self.signature = digest.hexdigest()

    def get_vocab(self):
        return self.backend.get_vocab()

    def decode(self, ids):
        return self.backend.decode(list(map(int, ids)), skip_special_tokens=False)

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def messages(self, messages, *, assistant_only=False, last_only=False, tools=None, template_kwargs=None):
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or "role" not in m or "content" not in m for m in messages):
            raise ValueError("messages must be a nonempty list of role/content mappings")
        source = self.template
        if isinstance(source, dict):
            source = source.get("tool_use" if tools is not None and "tool_use" in source else "default")
        if not isinstance(source, str):
            raise ValueError("Messages require an unambiguous chat template")
        if MARKER in json.dumps(messages, ensure_ascii=False):
            raise ValueError("Input contains a reserved template tracking marker")
        if isinstance(tools, str):
            tools = json.loads(tools)
        if isinstance(tools, list):
            tools = [json.loads(t) if isinstance(t, str) else t for t in tools]
        text = compiled_template(source).render(**{**self.special, **(template_kwargs or {}),
               "messages": messages, "tools": tools, "add_generation_prompt": False})
        spans, parts, offset, start = [], [], 0, None
        for item in re.split("("+re.escape(MARKER)+"(?:START|END)\ue001)", text):
            if item == MARKER+"START\ue001":
                if start is not None:
                    raise ValueError("Nested generation blocks are unsupported")
                start = offset
            elif item == MARKER+"END\ue001":
                if start is None:
                    raise ValueError("Unbalanced generation block")
                spans.append((start, offset))
                start = None
            else:
                parts.append(item)
                offset += len(item)
        assistants = [m for m in messages if m["role"] == "assistant"]
        weighted = any("loss" in m or "weight" in m for m in messages)
        if (assistant_only or last_only or weighted) and len(spans) != len(assistants):
            raise ValueError("Assistant masking and message weights require exactly one generation block per assistant message")
        for m in messages:
            if m["role"] != "assistant" and ("loss" in m or "weight" in m):
                raise ValueError("Message loss weights are supported only on assistant messages")
        encoding = self.backend.encode("".join(parts), add_special_tokens=False)
        weights = [0.0 if assistant_only or last_only else 1.0]*len(encoding.ids)
        span_weights = []
        for index in range(len(spans)):
            message = assistants[index] if index < len(assistants) else {}
            value = message.get("weight", message.get("loss", 1.0))
            weight = 1.0 if value is None else float(value)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Message loss weights must be finite and non-negative")
            if message.get("loss") is False:
                weight = 0.0
            if last_only and index != len(assistants)-1:
                weight = 0.0
            span_weights.append(weight)
        # A token crossing boundaries takes the last overlapping span's weight.
        # Search spans once per token instead of walking every token per message.
        starts = [left for left, _ in spans]
        for token, (a, b) in enumerate(encoding.offsets):
            index = bisect_left(starts, b)-1
            if index >= 0 and b > a and a < spans[index][1]:
                weights[token] = span_weights[index]
        return encoding.ids, weights
