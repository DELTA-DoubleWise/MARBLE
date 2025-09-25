import json
from dataclasses import dataclass, field, asdict, is_dataclass, fields
import json, warnings
from typing import List, Optional
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime

@dataclass
class ZipformerConfig:
    feature_dim: int = 80
    output_downsampling_factor: int = 2
    num_encoder_layers: List[int] = field(default_factory=lambda: [2, 2, 3, 4, 3, 2])
    downsampling_factor: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 4, 2])
    encoder_dim: List[int] = field(default_factory=lambda: [192, 256, 384, 512, 384, 256])
    feedforward_dim: List[int] = field(default_factory=lambda: [576, 768, 1152, 1536, 1152, 768])

    warmup_batches: float = 4000.0
    dropout: Optional[float] = None

    num_heads: List[int] = field(default_factory=lambda: [4, 4, 4, 8, 4, 4])
    query_head_dim: List[int] = field(default_factory=lambda: [32])
    value_head_dim: List[int] = field(default_factory=lambda: [12])
    pos_head_dim: List[int] = field(default_factory=lambda: [4])
    pos_dim: int = 48

    encoder_unmasked_dim: List[int] = field(default_factory=lambda: [192, 192, 256, 256, 256, 192])
    cnn_module_kernel: List[int] = field(default_factory=lambda: [31, 31, 15, 15, 15, 31])
    causal: bool = False
    chunk_size: List[int] = field(default_factory=lambda: [16, 32, 64, -1])
    left_context_frames: List[int] = field(default_factory=lambda: [64, 128, 256, -1])
    
    @classmethod
    def from_preset(cls, preset: str):
        presets = {
            "base": {
                "encoder_dim": [192, 256, 384, 512, 384, 256],
                "feedforward_dim": [576, 768, 1152, 1536, 1152, 768],
                "num_encoder_layers": [2, 2, 3, 4, 3, 2],
            },
            "large": {
                "encoder_dim": [192, 256, 512, 768, 512, 256],
                "feedforward_dim": [576, 768, 1536, 2304, 1536, 768],
                "num_encoder_layers": [2, 2, 4, 5, 4, 2],
            }
        }
        if preset not in presets:
            raise ValueError(f"Unsupported preset '{preset}' for {cls.__name__}. Supported: {list(presets)}")

        base = presets[preset]
        return cls(**base)

    @classmethod
    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__

    @classmethod
    def to_json(self, json_path: str):
        """Save config to a JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    # @classmethod
    # def from_pretrained(cls, path):
    #     with open(f"{path}/config.json") as f:
    #         data = json.load(f)
    #     return cls(**data)
    
    @classmethod
    def from_pretrained(cls, path: str):
        with open(os.path.join(path, "config.json")) as f:
            raw_cfg = json.load(f)

        valid = {f.name for f in fields(cls)}
        cfg = {k: v for k, v in raw_cfg.items() if k in valid}

        unknown = set(raw_cfg) - valid
        if unknown:
            warnings.warn(f"Ignoring unrecognized config keys: {unknown}")

        return cls(**cfg)

    def save_config(self, output_dir: str):
        """
        Save this config dataclass instance to output_dir/config.json.

        If an identical config already exists, skip saving.
        If a different config exists, back it up with a timestamp before saving.
        """
        if not is_dataclass(self):
            raise TypeError("Config must be a dataclass instance")

        new_config = asdict(self)
        os.makedirs(output_dir, exist_ok=True)
        config_path = Path(output_dir) / "config.json"

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    existing_config = json.load(f)
                if existing_config == new_config:
                    logging.info(f"[save_config] Skipped saving. Config identical to existing one.")
                    return
                # Backup old config
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = config_path.with_name(f"config.{timestamp}.bak.json")
                shutil.move(config_path, backup_path)
                logging.info(f"[save_config] Existing config backed up to: {backup_path}")
            except Exception as e:
                logging.warning(f"[save_config] Could not compare with existing config: {e}. Proceeding to save.")

        # Save new config
        with open(config_path, "w") as f:
            json.dump(new_config, f, indent=2)
        logging.info(f"[save_config] Saved config to: {config_path}")
