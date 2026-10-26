import logging
import sys
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-34s | rid=%(request_id)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY_LOGGERS = ("httpx", "httpcore", "mcp", "groq")

_base_record_factory = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs) -> logging.LogRecord:
    """Builds a log record enriched with the current request id."""
    record = _base_record_factory(*args, **kwargs)
    record.request_id = request_id_ctx.get()
    return record


def setup_logging(level_name: str) -> None:
    """Configures process-wide logging with request-id correlation.

    Every log line carries the request id of the request that produced it, so a
    single request can be traced across routers, controllers and graph nodes.

    Args:
        level_name: Log level name such as "INFO" or "DEBUG".
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.setLogRecordFactory(_record_factory)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
