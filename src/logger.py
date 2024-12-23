import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)