from copy import copy
import datadreamer.llms.openai

from contextlib import contextmanager
from unittest.mock import patch

_old_normalize_model_name = copy(datadreamer.llms.openai._normalize_model_name)


def _normalize_model_name(model_name):
    if "gpt-4o" in model_name:
        return "gpt-4-0125-preview"
    else:
        return _old_normalize_model_name(model_name)


@contextmanager
def datadreamer_gpt4o_support():
    # DataDreamer doesn't support gpt-4o yet (it will soon), but for now, we can apply a quick patch to make the model supported.
    with patch("datadreamer.llms.openai._normalize_model_name", _normalize_model_name):
        yield
