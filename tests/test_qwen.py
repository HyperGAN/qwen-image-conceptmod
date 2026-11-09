"""Qwen backend registry and convert-path smoke (no 20B weight load)."""
from conceptmod.providers import BackendRegistry
from qwen_image_conceptmod.provider import PROVIDER

REGISTRY = BackendRegistry([PROVIDER], discover=False)
REGISTERED = REGISTRY.names()
load_backend = REGISTRY.load
from qwen_image_conceptmod.backend import DEFAULT_MODEL, _LORA_TARGETS


def test_qwen_is_registered():
    assert "qwen" in REGISTERED
    assert DEFAULT_MODEL == "Qwen/Qwen-Image"
    assert _LORA_TARGETS == ["to_q", "to_k", "to_v", "to_out.0"]


def test_unknown_backend_still_lists_qwen():
    try:
        load_backend("not-a-backend", device="cuda:0")
    except LookupError as exc:
        assert "not-a-backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")
