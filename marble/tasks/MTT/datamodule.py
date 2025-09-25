# marble/tasks/MTT/datamodule.py
from typing import Optional

import torch

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset


class _MTTAudioBase(BaseAudioDataset):
    LABEL2IDX = {
        'guitar': 0,
        'classical': 1,
        'slow': 2,
        'techno': 3,
        'strings': 4,
        'drums': 5,
        'electronic': 6,
        'rock': 7,
        'fast': 8,
        'piano': 9,
        'ambient': 10,
        'beat': 11,
        'violin': 12,
        'vocal': 13,
        'synth': 14,
        'female': 15,
        'indian': 16,
        'opera': 17,
        'male': 18,
        'singing': 19,
        'vocals': 20,
        'no vocals': 21,
        'harpsichord': 22,
        'loud': 23,
        'quiet': 24,
        'flute': 25,
        'woman': 26,
        'male vocal': 27,
        'no vocal': 28,
        'pop': 29,
        'soft': 30,
        'sitar': 31,
        'solo': 32,
        'man': 33,
        'classic': 34,
        'choir': 35,
        'voice': 36,
        'new age': 37,
        'dance': 38,
        'male voice': 39,
        'female vocal': 40,
        'beats': 41,
        'harp': 42,
        'cello': 43,
        'no voice': 44,
        'weird': 45,
        'country': 46,
        'metal': 47,
        'female voice': 48,
        'choral': 49
        }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/MTT/mp3/0/american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-117-146.mp3", 
        "label": ["classical", "violin"], 
        "duration": 29.29375, 
        "sample_rate": 16000, 
        "num_samples": 468700, 
        "bit_depth": None, 
        "bitrate": 32034, 
        "channels": 1
    }


    def __init__(self, jsonl: str, sample_rate: int, channels: int,
                 clip_seconds: float, channel_mode: str="first",
                 min_clip_ratio: float=1.0, backend: Optional[str] = None):
        super().__init__(
            jsonl=jsonl,
            sample_rate=sample_rate,
            channels=channels,
            clip_seconds=clip_seconds,
            channel_mode=channel_mode,
            min_clip_ratio=min_clip_ratio,
            backend=backend
        )

    def get_targets(self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int):
        info = self.meta[file_idx]
        label_indices = [self.LABEL2IDX[tag] for tag in info['label']]
        num_labels = len(self.LABEL2IDX)
        label = torch.zeros(num_labels, dtype=torch.int)
        if len(label_indices) == 0:
            # If no labels, return a zero vector
            return label
        label[label_indices] = 1
        return label


class MTTAudioTrain(_MTTAudioBase):
    pass


class MTTAudioVal(_MTTAudioBase):
    pass


class MTTAudioTest(MTTAudioVal):
    pass


class MTTDataModule(BaseDataModule):
    pass