import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


def get_handlers(log_path):
    # check of log_path exists
    if not os.path.exists("logs/"):
        # create log_path folder
        os.makedirs("logs/")

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.ERROR)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    f_handler = TimedRotatingFileHandler(log_path, when="W1", backupCount=2)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)

    return [f_handler, c_handler]
