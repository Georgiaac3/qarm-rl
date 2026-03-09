import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="robot.log",
    filemode="w",
)
logger = logging.getLogger(__name__)
