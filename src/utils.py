import os
import logging
import functools
import numpy as np 

# Calculate the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_project_dir() -> str:
    """Return the project root directory."""
    return PROJECT_ROOT

def configure_logging() -> None:
    """Configure logging for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def singleton(cls):
    """Singleton decorator to ensure only one instance of a class exists."""
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
