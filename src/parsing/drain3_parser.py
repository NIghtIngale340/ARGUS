import base64
from pathlib import Path
import zlib

import jsonpickle
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class LogParser:
    def __init__(
        self,
        config_path: str = "configs/drain3.ini",
        state_path: str = "data/drain3_state.bin",
    ):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)

        config = TemplateMinerConfig()
        if self.config_path.exists():
            config.load(str(self.config_path))

        if isinstance(config.parameter_extraction_cache_capacity, str):
            config.parameter_extraction_cache_capacity = int(config.parameter_extraction_cache_capacity)

        self.miner = TemplateMiner(
            persistence_handler=None,
            config=config,
        )

    def parse(self, log_line: str) -> tuple[int, list[str]]:
        result = self.miner.add_log_message(log_line)

        template_id = int(result["cluster_id"])
        template = result["template_mined"]

        extracted = self.miner.extract_parameters(template, log_line) or []
        params = [p.value for p in extracted]

        return template_id, params

    def save_state(self, path: str | None = None) -> None:
        target = Path(path) if path else self.state_path
        target.parent.mkdir(parents=True, exist_ok=True)

        state_bytes = jsonpickle.dumps(self.miner.drain, keys=True).encode("utf-8")
        if self.miner.config.snapshot_compress_state:
            state_bytes = base64.b64encode(zlib.compress(state_bytes))

        with target.open("wb") as f:
            f.write(state_bytes)

    def load_state(self, path: str | None = None) -> bool:
        target = Path(path) if path else self.state_path
        if not target.exists():
            return False

        with target.open("rb") as f:
            state_bytes = f.read()

        if self.miner.config.snapshot_compress_state:
            state_bytes = zlib.decompress(base64.b64decode(state_bytes))

        loaded_drain = jsonpickle.loads(state_bytes, keys=True)
        self.miner.drain.__dict__.update(loaded_drain.__dict__)

        return True

    def get_template_count(self) -> int:
        return len(self.miner.drain.clusters)
