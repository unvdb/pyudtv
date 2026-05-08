import logging.config
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    MaxVarCharLengthKey = "max_length"
    EncodeProtocol = "utf-8"


# logging
COLORS = {
    "HEADER": "\033[95m",
    "INFO": "\033[92m",
    "DEBUG": "\033[94m",
    "WARNING": "\033[93m",
    "ERROR": "\033[95m",
    "CRITICAL": "\033[91m",
    "ENDC": "\033[0m",
}


class ColorFulFormatColMixin:
    def format_col(self, message_str: str, level_name: str):
        if level_name in COLORS:
            message_str = COLORS.get(level_name) + message_str + COLORS.get("ENDC")
        return message_str


class ColorfulFormatter(logging.Formatter, ColorFulFormatColMixin):
    def format(self, record: str):
        message_str = super().format(record)
        return self.format_col(message_str, level_name=record.levelname)


def init_log(log_level: str):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s][%(funcName)s]: %(message)s (%(filename)s:%(lineno)s)",
            },
            "colorful_console": {
                "format": "%(asctime)s | %(levelname)s: %(message)s (%(filename)s:%(lineno)s) (%(process)s)",
                "()": ColorfulFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "colorful_console",
            },
            "no_color_console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "pymilvus": {"handlers": ["no_color_console"], "level": log_level, "propagate": False},
            "pymilvus.milvus_client": {
                "handlers": ["no_color_console"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(logging_config)


init_log("WARNING")
