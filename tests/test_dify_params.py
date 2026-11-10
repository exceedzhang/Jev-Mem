"""Unit tests for the pure parameter helpers of the Dify plugin.

``integrations/dify/plugin/utils/params.py`` is loaded by file path under a
unique module name, so these run in the normal CI suite without installing the
Dify plugin runtime and without shadowing the repository's own ``utils``
package.
"""

import importlib.util
import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / "integrations" / "dify" / "plugin"


def _load_params():
    name = "jev_dify_plugin_params"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _PLUGIN_DIR / "utils" / "params.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


params = _load_params()


def test_normalize_result_mode_defaults_and_case():
    assert params.normalize_result_mode(None) == "answer"
    assert params.normalize_result_mode("") == "answer"
    assert params.normalize_result_mode("   ") == "answer"
    assert params.normalize_result_mode("MEMORY") == "memory"
    assert params.normalize_result_mode(" Answer ") == "answer"


def test_normalize_result_mode_rejects_unknown_value():
    try:
        params.normalize_result_mode("xml")
    except ValueError as exc:
        assert "result_mode" in str(exc)
        assert "xml" in str(exc)
    else:
        raise AssertionError("expected the invalid mode to be rejected")


def test_normalize_response_format_defaults_and_case():
    assert params.normalize_response_format(None) == "both"
    assert params.normalize_response_format("") == "both"
    assert params.normalize_response_format("JSON") == "json"
    assert params.normalize_response_format("Text") == "text"


def test_normalize_response_format_rejects_unknown_value():
    try:
        params.normalize_response_format("xml")
    except ValueError as exc:
        assert "response_format" in str(exc)
    else:
        raise AssertionError("expected the invalid format to be rejected")


def test_normalize_sort_order_defaults_and_case():
    assert params.normalize_sort_order(None) == "time"
    assert params.normalize_sort_order("") == "time"
    assert params.normalize_sort_order("   ") == "time"
    assert params.normalize_sort_order("SIMILARITY") == "similarity"
    assert params.normalize_sort_order(" Time ") == "time"


def test_normalize_sort_order_rejects_unknown_value():
    try:
        params.normalize_sort_order("bogus")
    except ValueError as exc:
        assert "sort" in str(exc)
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected the invalid sort order to be rejected")