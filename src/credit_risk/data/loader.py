from pathlib import Path

import pandas as pd
import yaml


def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_path_from_config(config_path: str | Path, relative_path: str) -> Path:
    """
    Resolve a project path declared in YAML relative to the project root.

    Assumes the config file lives inside /configs and paths are declared
    relative to the project root.
    """
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    return (project_root / relative_path).resolve()


def load_raw_data(paths_config_path: str | Path) -> pd.DataFrame:
    """Load raw dataset using the path defined in paths.yaml."""
    config = load_yaml_config(paths_config_path)
    raw_data_path = _resolve_path_from_config(
        paths_config_path,
        config["paths"]["raw_data"],
    )
    return pd.read_csv(raw_data_path)


def save_processed_data(
    df: pd.DataFrame,
    paths_config_path: str | Path,
) -> None:
    """Save processed dataset to the path defined in paths.yaml."""
    config = load_yaml_config(paths_config_path)
    processed_data_path = _resolve_path_from_config(
        paths_config_path,
        config["paths"]["processed_data"],
    )
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    with processed_data_path.open("w", encoding="utf-8", newline="") as file:
        df.to_csv(file, index=False)
