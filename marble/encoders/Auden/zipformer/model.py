# Copyright 2024–2025  Tencent AILab (Author: Yiwen Shao)
# Adapted from: https://github.com/k2-fsa/icefall
# Apache License 2.0

from typing import Optional, Tuple, List

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder.zipformer import Zipformer2
from .encoder.subsampling import Conv2dSubsampling
from .utils.scaling import ScheduledFloat
from ..modeling_output import ZipformerEncoderOutput
from .utils.padding import make_pad_mask
import numpy as np


class ZipformerEncoderModel(nn.Module):
    """
    This is the base zipformer encoder model
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = self.construct_feature_extractor()
        # Initialize encoder embedding
        self.encoder_embed = Conv2dSubsampling(
            in_channels=config.feature_dim,
            out_channels=config.encoder_dim[0],
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        )
        # Initialize encoder
        self.encoder = Zipformer2(
            output_downsampling_factor=config.output_downsampling_factor,
            downsampling_factor=tuple(config.downsampling_factor),
            encoder_dim=config.encoder_dim,
            num_encoder_layers=config.num_encoder_layers,
            encoder_unmasked_dim=config.encoder_unmasked_dim,
            query_head_dim=config.query_head_dim,
            pos_head_dim=config.pos_head_dim,
            value_head_dim=config.value_head_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            cnn_module_kernel=config.cnn_module_kernel,
            pos_dim=config.pos_dim,
            dropout=config.dropout,
            warmup_batches=config.warmup_batches,
            causal=config.causal,
            chunk_size=tuple(config.chunk_size),
            left_context_frames=tuple(config.left_context_frames),
        )
        
    def set_batch_count(self, count):
        for m in self.modules():
            if hasattr(m, "batch_count"):
                m.batch_count = count
        
    def forward_encoder(self, x, x_lens):
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          outputs.encoder_out:
            Encoder output, of shape (N, T, C).
          outputs.encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        outputs = self.encoder(x, x_lens, src_key_padding_mask)
        
        return ZipformerEncoderOutput(
            encoder_out=outputs[0].permute(1, 0, 2), # (T, N, C) ->(N, T, C)
            encoder_out_lens=outputs[1],
            encoder_out_full=outputs[2].permute(0, 2, 1, 3) # (N_blocks, N, T, C)
        )
               
    def construct_feature_extractor(self):
        import kaldifeat
        opts = kaldifeat.FbankOptions()
        opts.device = torch.device("cpu")
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        fbank = kaldifeat.Fbank(opts)
        return fbank
        
    def extract_feature(self, input):
        """
        Extract features from input, which can be:
            - (x, x_lens): tuple of precomputed features (Tensor, Tensor)
            - List[str]: list of wav file paths
            - List[np.ndarray] or List[Tensor]: list of 1D mono waveforms

        Returns:
            features (Tensor): (N, T, C)
            feature_lens (Tensor): (N,)
        """
        import math
        import torchaudio
        import logging
        from torch.nn.utils.rnn import pad_sequence

        # Case 1: already precomputed
        if isinstance(input, tuple) and len(input) == 2:
            return input

        wavs = []
        # Case 2: list of file paths
        if isinstance(input, list) and isinstance(input[0], str):
            logging.info(f"Reading sound files: {input}")
            for f in input:
                wave, sample_rate = torchaudio.load(f)  # shape: (1 or 2, num_samples)
                data = wave[0]  # take first channel
                if sample_rate != 16000:
                    print(f"Warning: sample rate for {f} is {sample_rate}Hz. Resampling to 16kHz...")
                    data = torchaudio.functional.resample(data, orig_freq=sample_rate, new_freq=16000)
                wavs.append(data)

        # Case 3: list of waveforms (Tensor or np.ndarray)
        elif isinstance(input, list) and all(
            isinstance(x, (torch.Tensor, np.ndarray)) for x in input
        ):
            for i, data in enumerate(input):
                data = data.squeeze(0)
                if isinstance(data, np.ndarray):
                    data = torch.tensor(data, dtype=torch.float32)
                if data.ndim != 1:
                    raise ValueError(f"Waveform at index {i} must be 1D mono, but got shape {data.shape}")
                wavs.append(data)

        else:
            raise ValueError(
                "Input must be either (x, x_lens), List[str] of wav paths, or List[1D waveform arrays]."
            )

        # Extract features
        features = self.feature_extractor(wavs)  # returns List[Tensor]
        feature_lens = [f.size(0) for f in features]
        features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
        feature_lens = torch.tensor(feature_lens, dtype=torch.long)

        return features, feature_lens