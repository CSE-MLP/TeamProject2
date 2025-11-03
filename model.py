from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModel


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

class OIDBertClassification(nn.Module):
    def __init__(
      self,
      num_labels = 3,
      dropout = 0.0,
      model_name = "microsoft/deberta-v3-small",
      lora_cfg = None,
      quant_config = None

    ) -> None:
        super().__init__()
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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)

        logits = self.classifier(cls_emb)
        return logits