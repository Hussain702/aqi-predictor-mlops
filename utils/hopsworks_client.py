"""
Shared helper for connecting to Hopsworks.

"""

import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")


def _ensure_tmp_dir_exists():
  
    os.makedirs("/tmp", exist_ok=True)


def get_feature_store():
    """
    Log into Hopsworks and return the Feature Store object for your project.
    """
    _ensure_tmp_dir_exists()

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    return project.get_feature_store()
