import torch
from torch.utils.data import Dataset
class CombineThreeSentencesDataset(Dataset):
    def __init__(self,df,tokenizer,max_length=170):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 문자열/라벨 준비
        self.ids = df.id.astype(int).tolist()
        self.prompts = df.prompt.astype(str).tolist()
        self.resp_a = df.response_a.astype(str).tolist()
        self.resp_b = df.response_b.astype(str).tolist()
        self.labels = torch.tensor(df.winner_model_b + 2*df.winner_tie, dtype=torch.long)

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

def pad_collate_fn(batch, pad_id):
    # batch: list of (input_ids, label)
    ids_list, labels = zip(*batch)
    lengths = [t.size(0) for t in ids_list]
    max_len = max(lengths)

    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)

    for i, t in enumerate(ids_list):
        L = t.size(0)
        input_ids[i, :L] = t
        attention_mask[i, :L] = 1

    labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, labels