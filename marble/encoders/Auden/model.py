from typing import Sequence, Optional, Tuple
from pathlib import Path

import torch

from marble.core.base_encoder import BaseEncoder

from marble.encoders.Auden.zipformer.model import ZipformerEncoderModel
from marble.encoders.Auden.zipformer.model_config import ZipformerConfig
from marble.encoders.Auden.utils.checkpoint import load_model_params

import pdb
import os, json


class Auden_Encoder(BaseEncoder):
    """Wrapper around Auden's Zipformer encoder for use in downstream tasks.

    This class mirrors the interface of encoders used in MARBLE, accepting raw
    waveforms and returning a tuple of hidden states.
    """

    NAME = "AudenZipformer"
    TOKEN_RATE = 25  # feature frames per second after subsampling
    SAMPLING_RATE = 16000
    NUM_FEATURES = 512  # hidden dimension of the last encoder stack
    N_TRANSFORMER_LAYERS = 16  # total number of transformer layers

    def __init__(
        self,
        pre_trained_folder: Optional[str] = None,
        checkpoint: Optional[str] = None,
        train_mode: str = "freeze",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
    ) -> None:
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE
        
        config_file = os.path.join(pre_trained_folder, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)

        if pre_trained_folder is not None:
            config = ZipformerConfig.from_pretrained(pre_trained_folder)
        else:
            config = ZipformerConfig.from_preset("base")

        self.model = ZipformerEncoderModel(config)
        # enable feature extraction since external code provides waveforms
        self.model.feature_extractor = self.model.construct_feature_extractor()

        init_modules = ['encoder_embed', 'encoder']

        if pre_trained_folder is not None:
            ckpt = Path(pre_trained_folder) / checkpoint
            if ckpt.is_file():
                load_model_params(self.model, ckpt, init_modules)

        if train_mode == "freeze":
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        elif train_mode == "full":
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()
        elif train_mode == "lora":
            raise NotImplementedError("LoRA adapters are not yet supported for Auden encoder")
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def forward(
        self,
        x: torch.Tensor,
        *args,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Encode a batch of raw waveforms.

        Args:
            x: Tensor of shape (batch_size, num_samples) with values in [-1, 1].
            output_hidden_states: If True, return intermediate block outputs in
                addition to the final representation.
            *args, **kwargs: Unused, kept for interface compatibility.

        Returns:
            Tuple of hidden states. The last element is always the final encoder
            representation of shape (batch_size, frames, ``NUM_FEATURES``).
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # convert batch tensor to list of 1D tensors on CPU for feature extraction
        wav_list = [w.detach().cpu() for w in x]
        feats, feat_lens = self.model.extract_feature(wav_list)

        feats = feats.to(device=device, dtype=dtype)
        feat_lens = feat_lens.to(device=device)

        enc_out = self.model.forward_encoder(feats, feat_lens)

        if output_hidden_states:
            hidden = tuple(enc_out.encoder_out_full) + (enc_out.encoder_out,)
        else:
            hidden = (enc_out.encoder_out,)
        return hidden


