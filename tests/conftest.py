import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(4)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False


@pytest.fixture
def tiny_config():
    from causal_tuner.modeling.config import ModelConfig
    return ModelConfig.tiny()


@pytest.fixture
def args():
    from causal_tuner.arguments import parse_args
    return parse_args(["--repo-id", "fixture", "--dataset-name", "fixture", "--dtype", "float32",
                       "--param-dtype", "float32", "--lora-rank", "4", "--attn-mechanism", "vanilla",
                       "--max-sequence-length", "9", "--loss-token-budget", "3", "--no-shuffle"])


@pytest.fixture
def tokenizer(tmp_path):
    import json
    from tokenizers import Tokenizer, models, pre_tokenizers
    from causal_tuner.data.tokenizer import TrainingTokenizer
    directory = tmp_path/"tokenizer"
    directory.mkdir()
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "user": 3, "assistant": 4,
                  "one": 5, "two": 6, "three": 7, "anchor": 8, "<s>": 9, "</s>": 10}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.save(str(directory/"tokenizer.json"))
    template = "{% for m in messages %}{{ m.role }} {% if m.role == 'assistant' %}{% generation %}{{ m.content }} </s> {% endgeneration %}{% else %}{{ m.content }} </s> {% endif %}{% endfor %}"
    (directory/"tokenizer_config.json").write_text(json.dumps({"pad_token": "[PAD]", "eos_token": "[EOS]", "chat_template": template}))
    return TrainingTokenizer(directory)


def batch(ids, weights=None, segments=None):
    ids = torch.as_tensor(ids, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "loss_weights": torch.ones_like(ids, dtype=torch.float32) if weights is None else torch.tensor(weights),
            "segment_ids": torch.zeros_like(ids) if segments is None else torch.as_tensor(segments)}
