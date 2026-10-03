import logging
import sys
from contextvars import ContextVar

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id_var.get()
        return True

def setup_logging():
    logger = logging.getLogger("prodml")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","message":"%(message)s",'
        '"correlation_id":"%(correlation_id)s"}'
    )
    handler.setFormatter(formatter)
    handler.addFilter(CorrelationIdFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()
