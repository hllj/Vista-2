import logging
from typing import Any
from types import SimpleNamespace
from datadreamer.llms import Anthropic
from functools import cached_property

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
    stop_after_attempt,
)

from datadreamer.utils.import_utils import ignore_litellm_warnings

class CustomAnthropic(Anthropic):

    @cached_property
    def retry_wrapper(self):
        with ignore_litellm_warnings():
            from litellm.exceptions import (
                APIConnectionError,
                APIError,
                RateLimitError,
                ServiceUnavailableError,
            )

        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential(multiplier=1, min=10, max=60),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(ServiceUnavailableError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        # @retry(
        #     retry=retry_if_exception_type(APIError),
        #     wait=wait_exponential(multiplier=1, min=3, max=300),
        #     before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
        #     after=after_log(tenacity_logger, logging.INFO),
        #     stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
        #     reraise=True,
        # )
        def _retry_wrapper(func, **kwargs):
            return func(**kwargs)

        _retry_wrapper.__wrapped__.__module__ = None  # type: ignore[attr-defined]
        _retry_wrapper.__wrapped__.__qualname__ = f"{self.__class__.__name__}.run"  # type: ignore[attr-defined]
        return _retry_wrapper

    @cached_property
    def client(self) -> Any:
        with ignore_litellm_warnings():
            from litellm import completion

        def wrapped_completion(*args, **kwargs):
            with ignore_litellm_warnings():
                from litellm.exceptions import (
                    BadRequestError,
                    APIConnectionError,
                    APIError,
                )
            try:
                return completion(*args, **kwargs)
            except (BadRequestError, APIConnectionError, APIError):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="I will not provide any information or assistance related to that."
                            )
                        )
                    ]
                )

        
        return wrapped_completion
