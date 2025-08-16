import pandas as pd
import torch
from transformers import EsmConfig,EsmTokenizer, EsmForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch.nn.functional as F
import wandb
import math
"""
wandb.login()  
wandb.init(
            project="mlm_pt_phosphosites", 
            name="esm1b_unlabaled_phosphosites_all",
            entity="entity_name",
        )  
"""

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    # Tensöre çevirelim (Hugging Face bazen numpy döndürebiliyor)
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

df = pd.read_csv("unlabeled_data_with_longer_form_128.csv")
sequences = df["Longer_Peptide"].tolist()

model_name = "facebook/esm1b_t33_650M_UR50S"
#model = EsmForMaskedLM.from_pretrained(model_name)

config = EsmConfig.from_pretrained(model_name)  
model = EsmForMaskedLM(config)
tokenizer = EsmTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=128)

full_dataset = Dataset.from_pandas(pd.DataFrame({"sequence": sequences}))
splitted = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = splitted["train"]
val_dataset = splitted["test"]

train_tokens = train_dataset.map(
    lambda examples: tokenizer(examples["sequence"], truncation=True, padding=True,max_length=128),
    batched=True
).shuffle(seed=42)

val_tokens = val_dataset.map(
    lambda examples: tokenizer(examples["sequence"], truncation=True, padding=True,max_length=128),
    batched=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15  
)


training_args = TrainingArguments(
    output_dir="./esm1b_mlm_pt_phosphosites_all",
    overwrite_output_dir=True,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    save_steps=10000,
    save_total_limit=3,
    num_train_epochs=100,
    fp16=True,
    logging_steps=500,  
    #report_to="wandb",  
    optim="adamw_torch",
    evaluation_strategy="steps",
    eval_steps=5000,
    load_best_model_at_end=True,
    run_name ="unlabeled_phosphosite_pt_128long",
    metric_for_best_model="perplexity",
    greater_is_better=False 
)

wandb.watch(model, log="all")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokens,
    eval_dataset=val_tokens,
    compute_metrics=compute_metrics
)

trainer.train()

## trainer.train(resume_from_checkpoint=True)