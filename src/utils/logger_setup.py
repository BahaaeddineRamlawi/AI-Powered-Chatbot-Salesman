import os
import logging
from datetime import datetime

from .config_loader import config

def setup_logging(config):
    """Sets up logging based on the config file."""
    logs_dir = config['logging']['logs_dir']
    os.makedirs(logs_dir, exist_ok=True)

    log_file = f"{logs_dir}/app_log_{datetime.now().strftime('%Y-%m-%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="a",
    )
    return logging.getLogger(__name__)

logging = setup_logging(config)