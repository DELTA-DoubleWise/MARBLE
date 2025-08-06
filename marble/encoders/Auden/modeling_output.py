from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AsrLossComponents:
    simple_loss: Optional[float] = None
    pruned_loss: Optional[float] = None
    ctc_loss: Optional[float] = None
    attention_decoder_loss: Optional[float] = None
    lid_loss: Optional[float] = None
    balance_loss: Optional[float] = None


@dataclass
class IcefallAsrModelOutput:
    loss: AsrLossComponents = field(default_factory=AsrLossComponents)
    lid_output: Optional[Any] = None
    gate_logits: Optional[Any] = None
    padding_mask: Optional[Any] = None
    
@dataclass
class ZipformerEncoderOutput:
    encoder_out: Optional[Any] = None
    encoder_out_lens: Optional[Any] = None
    encoder_out_full: Optional[Any] = None