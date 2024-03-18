import logging


class ColorCodes:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S,%f"[:-3]  # Date format, microseconds trimmed

    COLORS = {
        logging.DEBUG: ColorCodes.GREEN,
        logging.INFO: ColorCodes.RESET,
        logging.WARNING: ColorCodes.YELLOW,
        logging.ERROR: ColorCodes.RED,
        logging.CRITICAL: ColorCodes.RED,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, ColorCodes.RESET)
        record.levelname = f"{color}{record.levelname}{ColorCodes.RESET}"
        record.msg = f"{color}{record.msg}{ColorCodes.RESET}"
        return super().format(record)


def setup_global_logger(logger, file, level=logging.INFO):
    # color formatter
    formatter = ColorFormatter(ColorFormatter.FORMAT)

    # steaming handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # file handler
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)

    logger.handlers = [file_handler, handler]
    logger.setLevel(level)
