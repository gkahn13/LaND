from loguru import logger
import sys


def setup(log_fname=None, exp_name=''):
    logger.remove()
    if log_fname:
        logger.add(log_fname,
                   format=exp_name + " {time} {level} {message}",
                   level="DEBUG")
    logger.add(sys.stdout,
               colorize=True,
               format="<yellow>" + exp_name + "</yellow> | "
                                              "<green>{time:HH:mm:ss}</green> | "
                                              "<blue>{level: <8}</blue> | "
                                              "<magenta>{name}:{function}:{line: <5}</magenta> | "
                                              "<white>{message}</white>",
               level="DEBUG",
               filter=lambda record: record["level"].name == "DEBUG")
    logger.add(sys.stdout,
               colorize=True,
               format="<yellow>" + exp_name + "</yellow> | "
                                              "<green>{time:HH:mm:ss}</green> | "
                                              "<blue>{level: <8}</blue> | "
                                              "<white>{message}</white>",
               level="INFO")

def debug(s):
    logger.debug(s)

def info(s):
    logger.info(s)

def warning(s):
    logger.warning(s)