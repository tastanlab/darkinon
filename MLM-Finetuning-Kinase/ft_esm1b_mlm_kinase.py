## slurm'e eklenecek
"""
!pip install transformers accelerate wandb fair-esm
!pip install datasets
!wandb login
"""

import pickle
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    EsmForMaskedLM,
    EsmConfig,
    EsmTokenizer
    TrainingArguments
)
import random
from datasets import Dataset
import math
import torch.nn.functional as F
import wandb

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    B, L, V = logits.size()
    logits = logits.view(B * L, V)
    labels = labels.view(B * L)

    mask = labels != -100
    logits_masked = logits[mask]
    labels_masked = labels[mask]

    ce_loss = F.cross_entropy(logits_masked, labels_masked)
    perplexity = math.exp(ce_loss.item())
    return {
        "perplexity": perplexity
    }


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

model_name = "facebook/esm1b_t33_650M_UR50S"

model = EsmForMaskedLM.from_pretrained(model_name) 
tokenizer = EsmTokenizer.from_pretrained(model_name)

data = pickle.load(open("kinase_msa_all_200_sequence.pkl", "rb"))

protein_sequences = []
for actual, msa in data.items():

  for m in msa:
    idx, seq = m[0],m[1]
    protein_sequences.append(seq)

removed_gaps_protein_sequences = []
for seq in protein_sequences:
  removed_gaps_protein_sequences.append(seq.replace("-", ""))

print("Data size:", len(removed_gaps_protein_sequences))
print(removed_gaps_protein_sequences[0])
full_dataset = Dataset.from_dict({"text": removed_gaps_protein_sequences})

splitted = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = splitted["train"]
val_dataset = splitted["test"]

train_tokens = train_dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, padding=True),
    batched=True
)

val_tokens = val_dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, padding=True),
    batched=True
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


wandb.init(
            project="mlm_kinase", 
            name="ft_kinase_msa_mlm_esm1b",
            entity="entity_name",
        )

training_args = TrainingArguments(
    output_dir="./esm1b-mlm-ft-kinase-checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    save_steps=10000,
    save_total_limit=3,
    num_train_epochs=100,
    bf16=True,
    logging_steps=500,  # 500 adımda bir log kaydedelim
    optim="adamw_torch",
    evaluation_strategy="steps",
    eval_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="perplexity",
    greater_is_better=False # "perplexity"yi en iyi model ölçütü olarak al
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokens,
    eval_dataset=val_tokens,
    compute_metrics=compute_metrics
)

trainer.train()

model.push_to_hub("xxxx", use_auth_token="xxxx")
tokenizer.push_to_hub("xxxx", use_auth_token="xxxx")