"""M4: tokenizer extension, embedding resize, LoRA-only training, generation format."""

import torch
import pytest

from model.backbone import assert_lora_only, load_model, trainable_summary
from model.config import CODEBOOK_SIZE, N_ACTION_TOKENS, BackboneConfig
from model.inference import generate_plan
from model.prompting import parse_response
from model.standin import build_tiny_backbone

CFG = BackboneConfig(load_in_4bit=False)
MSGS = [{"role": "system", "content": "You are an autonomous driving planner."},
        {"role": "user", "content": [{"type": "text", "text": "Navigation instruction: Stop."}]}]


@pytest.fixture(scope="module")
def bb():
    model, tok = build_tiny_backbone()
    return load_model(CFG, model=model, tokenizer=tok)


def test_action_tokens_contiguous(bb):
    ids = bb.action_token_ids
    assert len(ids) == CODEBOOK_SIZE and ids == list(range(ids[0], ids[0] + CODEBOOK_SIZE))
    assert bb.tokenizer.convert_ids_to_tokens(ids[7]) == "<action_7>"
    assert bb.tokenizer("<action_5><action_6>", add_special_tokens=False)["input_ids"] == [ids[5], ids[6]]


def test_embeddings_cover_vocab(bb):
    emb = bb.model.get_input_embeddings().weight
    assert emb.shape[0] >= len(bb.tokenizer) and emb.shape[0] % 64 == 0


def test_only_lora_and_new_token_rows_train(bb):
    s = assert_lora_only(bb.model)
    hidden = bb.model.get_input_embeddings().weight.shape[1]
    assert s["lora"] > 0 and s["vision"] == 0 and s["other"] == 0
    assert s["new_tokens"] == CODEBOOK_SIZE * hidden
    assert not any(p.requires_grad for n, p in bb.model.named_parameters() if "visual" in n)


def test_lora_targets_exclude_vision(bb):
    names = [n for n, _ in bb.model.named_parameters() if "lora_A" in n]
    assert names and not any("visual" in n for n in names)


def test_new_token_rows_get_gradient_through_tied_head(bb):
    ids = torch.tensor([bb.tokenizer("Stop.", add_special_tokens=False)["input_ids"] + bb.action_token_ids[:3]])
    labels = ids.clone(); labels[:, :-3] = -100
    bb.model.zero_grad()
    bb.model(input_ids=ids, labels=labels).loss.backward()
    deltas = [p for n, p in bb.model.named_parameters() if "trainable_tokens" in n]
    assert deltas and all(p.grad is not None and p.grad.abs().sum() > 0 for p in deltas)


def test_untied_head_trains_its_own_rows():
    model, tok = build_tiny_backbone(tie_word_embeddings=False)
    b = load_model(CFG, model=model, tokenizer=tok)
    s = assert_lora_only(b.model)
    hidden = b.model.get_input_embeddings().weight.shape[1]
    assert s["new_tokens"] == 2 * CODEBOOK_SIZE * hidden


@pytest.mark.parametrize("mode", ["fast", "slow", "auto"])
def test_generation_always_emits_ten_valid_actions(bb, mode):
    plan = generate_plan(bb, MSGS, mode=mode, do_sample=True, temperature=1.5, cot_max_tokens=12)
    assert plan.valid and len(plan.action_ids) == N_ACTION_TOKENS
    assert all(0 <= i < CODEBOOK_SIZE for i in plan.action_ids)
    if mode == "fast":
        assert plan.n_cot_tokens == 0 and parse_response(plan.text).mode == "fast"
    if mode == "slow":
        assert plan.n_cot_tokens <= 12 and parse_response(plan.text).mode == "slow"
    assert plan.mode in ("fast", "slow")


def test_vision_path_runs_and_is_frozen(bb):
    tok = bb.tokenizer
    t = lambda s: tok.convert_tokens_to_ids(s)
    ids = [t("<|vision_start|>")] + [t("<|image_pad|>")] * 4 + [t("<|vision_end|>")] \
        + tok("Stop.", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([ids])
    pix = torch.randn(16, 3 * 2 * 14 * 14)           # one 4x4-patch image -> 4 merged tokens
    out = bb.model(input_ids=ids, pixel_values=pix, image_grid_thw=torch.tensor([[1, 4, 4]]))
    assert out.logits.shape[:2] == ids.shape and torch.isfinite(out.logits).all()


def test_load_base_forwards_expected_kwargs(monkeypatch):
    """The real-model loader can't run here (no weights/CUDA); check its plumbing with fakes."""
    import transformers
    from model.backbone import load_base

    seen = {}
    tiny_model, tok = build_tiny_backbone()

    class FakeProc:
        tokenizer = tok

    monkeypatch.setattr(transformers.Qwen2_5_VLForConditionalGeneration, "from_pretrained",
                        classmethod(lambda cls, mid, **kw: seen.update(model_id=mid, **kw) or tiny_model))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        classmethod(lambda cls, mid, **kw: seen.update(proc_kw=kw) or FakeProc()))
    model, proc = load_base(BackboneConfig(load_in_4bit=True))          # CPU: 4-bit is skipped with a notice
    assert model is tiny_model and proc.tokenizer is tok
    assert seen["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct" and "quantization_config" not in seen
    assert any(k in seen for k in ("dtype", "torch_dtype")) and seen["proc_kw"]["max_pixels"] > 0
