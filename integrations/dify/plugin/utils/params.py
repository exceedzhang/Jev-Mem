"""Pure validation helpers for the Jev-Mem plugin tool parameters.

This module is deliberately dependency-free (no Dify SDK imports, no imports
from the plugin's own client) so the helpers can be unit-tested without the
plugin runtime installed. ``tools/base.py`` adapts the ``ValueError`` raised
here to the plugin's ``JMemClientError`` type for the tool call sites.
"""

from typing import Any

RESPONSE_FORMATS = ("both", "json", "text")
RESULT_MODES = ("answer", "memory")
SORT_ORDERS = ("time", "similarity")


def _normalize(value: Any, default: str, allowed: tuple[str, ...], name: str) -> str:
    raw = "" if value is None else str(value).strip().lower()
    if not raw:
        # An optional form field the user cleared arrives as "" — treat it as
        # unset rather than as an invalid value.
        return default
    if raw not in allowed:
        raise ValueError(
            f"Parameter '{name}' must be one of {', '.join(allowed)}; got {value!r}."
        )
    return raw


def normalize_result_mode(value: Any) -> str:
    """Validate the ``result_mode`` tool parameter.

    ``answer`` (default) lets the sidecar synthesize an answer from the
    retrieved evidence, ``memory`` returns the raw memory records instead so
    the calling model can post-process them with its own instructions.
    """
    return _normalize(value, "answer", RESULT_MODES, "result_mode")


def normalize_response_format(value: Any) -> str:
    """Validate the ``response_format`` tool parameter.

    ``both`` (default) yields a JSON message and a text message, ``json``
    yields only the structured ``json`` output, ``text`` only the ``text``
    output.
    """
    return _normalize(value, "both", RESPONSE_FORMATS, "response_format")


def normalize_sort_order(value: Any) -> str:
    """Validate the ``sort`` tool parameter for the memory list.

    ``time`` (default) returns the memories newest-first; ``similarity``
    returns them in retrieval-relevance order.
    """
    return _normalize(value, "time", SORT_ORDERS, "sort")