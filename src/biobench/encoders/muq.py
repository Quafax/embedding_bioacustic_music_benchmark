from __future__ import annotations

import torch

from biobench.encoders.base import BaseEncoder, EncoderSpec


class MuQEncoder(BaseEncoder):
    """Adapter for the public MuQ API documented by Tencent AI Lab."""

    def __init__(self, spec: EncoderSpec, device: str = "auto") -> None:
        super().__init__(spec, device)
        try:
            from muq import MuQ
        except ImportError as exc:
            raise ImportError("Install MuQ support with: pip install -e '.[muq]'") from exc
        if not spec.model_id:
            raise ValueError("MuQ requires a model_id in the encoder registry.")
        self.model = MuQ.from_pretrained(spec.model_id).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Official MuQ example calls the model positionally and recommends fp32.
        output = self.model(audio.to(self.device, dtype=torch.float32), output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            layer = self.spec.layer if self.spec.layer is not None else -1
            return hidden_states[layer]
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("MuQ output has neither hidden_states nor last_hidden_state.")
        return last_hidden_state
