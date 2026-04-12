from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


def log_execution_time(func: Callable) -> Callable:
    """Log execution time and errors for a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            return result

        except Exception as e:
            logger.exception(
                "Function '%s' failed with error: %s",
                func.__name__,
                str(e),
            )
            raise  # vuelve a lanzar el error

        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            logger.info(
                "Function '%s' executed in %.4f seconds",
                func.__name__,
                elapsed,
            )

    return wrapper
