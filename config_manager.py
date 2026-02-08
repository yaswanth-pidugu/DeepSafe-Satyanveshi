import yaml
import os
from pathlib import Path


class ConfigManager:
    def __init__(self):
        # This finds the directory of THIS file (the root), no matter where you call it from
        self.root = Path(__file__).resolve().parent
        self.env = os.getenv("DEEPSAFE_ENV", "local")
        self.config = self._load_all()

    def _load_all(self):
        base_path = self.root / "configs" / "base.yaml"

        # Debugging: If it fails, it will tell you EXACTLY where it looked
        if not base_path.exists():
            raise FileNotFoundError(f"CRITICAL: Could not find {base_path}. Check your folder structure!")

        with open(base_path, "r") as f:
            base = yaml.safe_load(f)

        env_file = self.root / "configs" / f"{self.env}.yaml"
        if env_file.exists():
            with open(env_file, "r") as f:
                overrides = yaml.safe_load(f)
                self._deep_update(base, overrides)
        return base

    def _deep_update(self, base, overrides):
        for k, v in overrides.items():
            if isinstance(v, dict):
                base[k] = self._deep_update(base.get(k, {}), v)
            else:
                base[k] = v
        return base


cfg = ConfigManager().config