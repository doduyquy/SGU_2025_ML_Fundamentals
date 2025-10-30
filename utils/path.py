# This module provide a utility to get the assets path
from pathlib import Path

# -> Path: means this function returns a Path object
def get_assets_path() -> Path:
    """
    Get the absolute path to the assets directory.

    Returns:
        Path: The absolute path to the assets directory.
    """
    return Path.cwd().resolve().parent.parent.joinpath("assets")


def get_data_path() -> Path:
    """
    Get the absolute path to the datasets directory.

    Returns:
        Path: The absolute path to the datasets directory.
    """
    return Path.cwd().resolve().parent.parent.joinpath("datasets")