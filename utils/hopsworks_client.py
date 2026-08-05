"""
Shared helper for connecting to Hopsworks.
Used by etl/load.py now, and by training/ scripts later.
"""

import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")


def _ensure_tmp_dir_exists():
  
    os.makedirs("/tmp", exist_ok=True)


def get_project():
    """
    Log into Hopsworks and return the Project object (gives access to both
    the Feature Store and the Model Registry).
    """
    _ensure_tmp_dir_exists()

    return hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )


def get_feature_store():
    """Log into Hopsworks and return the Feature Store object for your project."""
    return get_project().get_feature_store()