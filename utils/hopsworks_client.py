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
    """
    Windows fix: the hopsworks library hard-codes '/tmp' as the folder where
    it stores connection certificates. On Linux/Mac that's a real folder.
    On Windows, '/tmp' (no drive letter) resolves to the root of the CURRENT
    DRIVE (e.g. D:\\tmp), and the library fails because that folder doesn't
    exist yet. Pre-creating it here fixes the error regardless of which
    drive the project is run from. This is a no-op (and harmless) on
    Linux/Mac, where /tmp already exists.
    """
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
