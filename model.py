from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers import MobileBertConfig, MobileBertModel

from peft import get_peft_model, prepare_model_for_kbit_training

class OIDBertClassification(nn.Module):
    def __init__(
      self,
      num_labels = 3,
      dropout = 0.1,
      pooling:str = "cls",
      model_name = "microsoft/deberta-v3-small",
      lora_cfg = None,
      quant_config = None

    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)

        if quant_config is not None:
            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quant_config,          # 4bit 사용 (bnb 가능 시)
                device_map="auto"
            )
            self.model = prepare_model_for_kbit_training(model)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        if lora_cfg is not None:
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()  # 디버그: LoRA 파라미터 수 확인
        
        hidden = self.config.hidden_size
        self.pooling = pooling.lower()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]  # [CLS]-like (actually first token)
        elif self.pooling == "max":
            # Mask padded positions to very negative before max
            mask = attention_mask.unsqueeze(-1).bool()
            masked = last_hidden_state.masked_fill(~mask, -1e9)
            return masked.max(dim=1).values
        else:  # mean (mask-aware)
            mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            return summed / denom

    def encode(self, **inputs) -> torch.Tensor:
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        if hasattr(self.model, "hf_device_map"):  # device_map="auto"로 샤딩된 경우
            # CPU 텐서를 그대로 넘기면 dispatch가 처리
            pass
        else:
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        h = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        z = self._pool(h, attention_mask)
        z = self.dropout(z)
        logits = self.classifier(z)
        return logits
    

class TIDBertClassification(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        pooling: str = "cls",  # one of {"cls", "mean", "max"}
        classifier_hidden: int = 384,
        dropout: float = 0.1,
        num_labels: int = 2,
        lora_cfg:bool = False,
        quant_config:bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        
        if quant_config is not None:
            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quant_config,          # 4bit 사용 (bnb 가능 시)
                device_map="auto"
            )
            self.model = prepare_model_for_kbit_training(model)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        if lora_cfg is not None:
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()  # 디버그: LoRA 파라미터 수 확인

        hidden = self.config.hidden_size
        self.pooling = pooling.lower()
        self.dropout = nn.Dropout(dropout)

        # Concatenate pooled A and B: 2 * hidden
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]  # [CLS]-like (actually first token)
        elif self.pooling == "max":
            # Mask padded positions to very negative before max
            mask = attention_mask.unsqueeze(-1).bool()
            masked = last_hidden_state.masked_fill(~mask, -1e9)
            return masked.max(dim=1).values
        else:  # mean (mask-aware)
            mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            return summed / denom

    def encode(self, **inputs) -> torch.Tensor:
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ):
        # Run shared backbone on A
        h_a = self.encode(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
        )
        # Run shared backbone on B
        h_b = self.encode(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
        )

        # Token-level -> sequence-level pooling
        z_a = self._pool(h_a, attention_mask_a)
        z_b = self._pool(h_b, attention_mask_b)

        # Concatenate and classify
        z = torch.cat([z_a, z_b], dim=-1)
        z = self.dropout(z)
        logits = self.classifier(z)

        return logits

class TIDMobileBertClassification(nn.Module):
    def __init__(
        self,
        pooling: str = "cls",  # one of {"cls", "mean", "max"}
        dropout: float = 0.1,
        num_labels: int = 3,
    ):
        super().__init__()
        self.config = MobileBertConfig()
        self.model = MobileBertModel(self.config)

        hidden = self.config.hidden_size
        self.pooling = pooling.lower()
        self.dropout = nn.Dropout(dropout)

        # Concatenate pooled A and B: 2 * hidden
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]  # [CLS]-like (actually first token)
        elif self.pooling == "max":
            # Mask padded positions to very negative before max
            mask = attention_mask.unsqueeze(-1).bool()
            masked = last_hidden_state.masked_fill(~mask, -1e9)
            return masked.max(dim=1).values
        else:  # mean (mask-aware)
            mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            return summed / denom

    def encode(self, **inputs) -> torch.Tensor:
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ):
        # Run shared backbone on A
        h_a = self.encode(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
        )
        # Run shared backbone on B
        h_b = self.encode(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
        )

        # Token-level -> sequence-level pooling
        z_a = self._pool(h_a, attention_mask_a)
        z_b = self._pool(h_b, attention_mask_b)

        # Concatenate and classify
        z = torch.cat([z_a, z_b], dim=-1)
        z = self.dropout(z)
        logits = self.classifier(z)

        return logits