from loguru import logger
from pathlib import Path

def setup_logger(path, rotation="10 MB"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    # file sink
    logger.add(str(p), rotation=rotation, enqueue=True, backtrace=False, diagnose=False)
    # console sink
    logger.add(lambda m: print(m, end=""), level="INFO")
    return logger
