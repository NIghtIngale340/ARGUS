import yaml
from pathlib import Path

def load_config(config_name: str) -> dict:
    if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
        config_name = f"{config_name}.yaml"
    config_path = Path("configs") / config_name
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)