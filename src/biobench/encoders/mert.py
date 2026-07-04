from __future__ import annotations

import torch

from biobench.encoders.base import BaseEncoder, EncoderSpec


class MERTEncoder(BaseEncoder):
    def __init__(self, spec: EncoderSpec, device: str = "auto") -> None:
        super().__init__(spec, device)
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install MERT support with: pip install -e '.[mert]'") from exc
        if not spec.model_id:
            raise ValueError("MERT requires a model_id in the encoder registry.")
        self.model = AutoModel.from_pretrained(
            spec.model_id,
            trust_remote_code=True,
        ).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        kwargs: dict[str, torch.Tensor | bool] = {
            "input_values": audio.to(self.device),
            "output_hidden_states": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self.device)
        output = self.model(**kwargs)
        hidden_states = getattr(output, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("MERT did not return hidden_states. Check transformers/MERT compatibility.")
        layer = self.spec.layer if self.spec.layer is not None else -1
        return hidden_states[layer]
