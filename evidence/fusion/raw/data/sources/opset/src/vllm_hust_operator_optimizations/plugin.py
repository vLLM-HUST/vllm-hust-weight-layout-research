from __future__ import annotations

import logging
import os

from .import_hook import install
from .registry import get_registry, register_builtins

logger = logging.getLogger(__name__)
ENV_NAME = "VLLM_HUST_OPERATOR_OPTIMIZATIONS"
_activated = False


def activate() -> None:
    """vLLM general-plugin entry point; safe to call once in every process."""
    global _activated
    if _activated:
        return
    register_builtins()

    configured = os.getenv(ENV_NAME, "").strip()
    if not configured:
        return
    enabled = [item.strip() for item in configured.split(",") if item.strip()]
    try:
        selected = get_registry().select(enabled)
        install(selected)
    except Exception as error:
        raise SystemExit(f"operator optimization activation rejected: {error}") from error
    _activated = True
    logger.info("Enabled operator optimizations: %s", ", ".join(enabled))
