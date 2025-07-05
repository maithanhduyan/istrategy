import logging
import logging.handlers
from queue import Queue

_log_queue = Queue(-1)
_listener_obj = None


def get_logger(name: str = None):
    """
    Get a logger with the specified name. Ensures no duplicate handlers.
    Uses asynchronous logging to file and console for performance.
    """
    global _listener_obj
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
        qh = logging.handlers.QueueHandler(_log_queue)
        logger.addHandler(qh)
        # Only set up the listener once
        if _listener_obj is None:
            file_handler = logging.FileHandler('./app.log', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            _listener_obj = logging.handlers.QueueListener(_log_queue, file_handler, console_handler)
            _listener_obj.start()
    return logger


def stop_logger():
    """
    Stop the QueueListener and flush all logs before exit.
    """
    global _listener_obj
    if _listener_obj is not None:
        _listener_obj.stop()
        _listener_obj = None

