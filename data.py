import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class OneInputDataset(Dataset):
    def __init__(self,df,tokenizer,max_length=170):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 문자열/라벨 준비
        self.ids = df.id.astype(int).tolist()
        self.prompts = df.prompt.astype(str).tolist()
        self.resp_a = df.response_a.astype(str).tolist()
        self.resp_b = df.response_b.astype(str).tolist()
        self.labels = torch.tensor(df.winner_model_b.astype(int) + 2*df.winner_tie.astype(int), dtype=torch.long)

        # 토큰 id
        self.cls_id = getattr(tokenizer, "cls_token_id", None)
        self.sep_id = getattr(tokenizer, "sep_token_id", None)
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_id = getattr(tokenizer, "pad_token_id", None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        p = self.tokenizer(self.prompts[idx],padding=True,truncation=True,max_length=self.max_length,add_special_tokens=False,return_tensors=None)
        a = self.tokenizer(self.resp_a[idx],padding=True,truncation=True,max_length=self.max_length,add_special_tokens=False,return_tensors=None)
        b = self.tokenizer(self.resp_b[idx],padding=True,truncation=True,max_length=self.max_length,add_special_tokens=False,return_tensors=None)

        ids = [self.cls_id] + p["input_ids"] + [self.sep_id] + a["input_ids"] + [self.sep_id] + b["input_ids"] + [self.eos_id]
        input_ids = torch.tensor(ids, dtype=torch.long)
        label = self.labels[idx]
        return input_ids, label

def pad_collate_fn_OID(batch, pad_token_id=0):
    # 각각의 요소를 분리
    input_ids_list, labels = zip(*batch)

    # 시퀀스 길이 맞추기 (패딩)
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)

    # attention mask 생성 (패딩이 아닌 부분 = 1)
    attention_mask = (input_ids_padded != pad_token_id).long()

    # 라벨 텐서로 묶기
    labels = torch.stack(labels)

    # 반환
    return input_ids_padded, attention_mask, labels


class TwoInputDataset(Dataset):
    def __init__(self,df,tokenizer,max_length=170):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 문자열/라벨 준비
        self.ids = df.id.astype(int).tolist()
        self.prompts = df.prompt.astype(str).tolist()
        self.resp_a = df.response_a.astype(str).tolist()
        self.resp_b = df.response_b.astype(str).tolist()
        self.labels = torch.tensor(df.winner_model_b.astype(int) + 2*df.winner_tie.astype(int), dtype=torch.long)

        # 토큰 id
        self.cls_id = getattr(tokenizer, "cls_token_id", None)
        self.sep_id = getattr(tokenizer, "sep_token_id", None)
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_id = getattr(tokenizer, "pad_token_id", None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        p = self.tokenizer(self.prompts[idx],padding=True,truncation=True,max_length=self.max_length,add_special_tokens=False,return_tensors=None)
        a = self.tokenizer(self.resp_a[idx],padding=True,truncation=True,max_length=self.max_length,add_special_tokens=False,return_tensors=None)
        b = self.tokenizer(self.resp_b[idx],padding=True,truncation=True,max_length=self.max_length,add_special_tokens=False,return_tensors=None)

        ids1 = [self.cls_id] + p["input_ids"] + [self.sep_id] + a["input_ids"] + [self.eos_id]
        ids2 = [self.cls_id] + p["input_ids"] + [self.sep_id] + b["input_ids"] + [self.eos_id]
        input_ids1 = torch.tensor(ids1, dtype=torch.long)
        input_ids2 = torch.tensor(ids2, dtype=torch.long)
        label = self.labels[idx]
        return input_ids1, input_ids2, label

class TwoInputDatasetv2(Dataset):
    def __init__(self,df,tokenizer,max_length=None):
        super().__init__()
        self.tokenizer = tokenizer

        # 문자열/라벨 준비
        self.ids = df.id.astype(int).tolist()
        self.prompts = df.prompt.astype(str).tolist()
        self.resp_a = df.response_a.astype(str).tolist()
        self.resp_b = df.response_b.astype(str).tolist()
        self.labels = torch.tensor(df.winner_model_b.astype(int) + 2*df.winner_tie.astype(int), dtype=torch.long)

        # 토큰 id
        self.cls_id = getattr(tokenizer, "cls_token_id", None)
        self.sep_id = getattr(tokenizer, "sep_token_id", None)
        self.pad_id = getattr(tokenizer, "pad_token_id", None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        p = self.tokenizer(self.prompts[idx],padding=True,truncation=True,max_length=69,add_special_tokens=False,return_tensors=None)
        a = self.tokenizer(self.resp_a[idx],padding=True,truncation=True,max_length=440,add_special_tokens=False,return_tensors=None)
        b = self.tokenizer(self.resp_b[idx],padding=True,truncation=True,max_length=440,add_special_tokens=False,return_tensors=None)

        ids1 = [self.cls_id] + p["input_ids"] + [self.sep_id] + a["input_ids"] + [self.sep_id]
        ids2 = [self.cls_id] + p["input_ids"] + [self.sep_id] + b["input_ids"] + [self.sep_id]
        input_ids1 = torch.tensor(ids1, dtype=torch.long)
        input_ids2 = torch.tensor(ids2, dtype=torch.long)
        label = self.labels[idx]
        return input_ids1, input_ids2, label

def pad_collate_fn_TID(batch, pad_token_id=0):
    # 각각의 요소를 분리
    input_ids1_list, input_ids2_list, labels = zip(*batch)

    # 시퀀스 길이 맞추기 (패딩)
    input_ids1_padded = pad_sequence(input_ids1_list, batch_first=True, padding_value=pad_token_id)
    input_ids2_padded = pad_sequence(input_ids2_list, batch_first=True, padding_value=pad_token_id)
    
    # attention mask 생성 (패딩이 아닌 부분 = 1)
    attention_mask1 = (input_ids1_padded != pad_token_id).long()
    attention_mask2 = (input_ids2_padded != pad_token_id).long()

    # 라벨 텐서로 묶기
    labels = torch.stack(labels)

    # 반환
    return input_ids1_padded, attention_mask1, input_ids2_padded, attention_mask2, labels
