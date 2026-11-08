import os
from dataclasses import dataclass


@dataclass
class Settings:
    data_dir: str = os.environ.get('DATA_DIR', 'data/local')


SETTINGS = Settings()
