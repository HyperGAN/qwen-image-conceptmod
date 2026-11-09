"""Lightweight installed-provider metadata; model code is loaded on creation."""
from conceptmod.contracts import BackendCapabilities
from conceptmod.providers import BackendProvider

PROVIDER = BackendProvider(
    name="qwen",
    factory="qwen_image_conceptmod.backend:QwenBackend",
    distribution="qwen-image-conceptmod",
    capabilities=BackendCapabilities(),
    training_defaults=(('sample_steps', 14), ('sample_guidance', 4.0)),
)
