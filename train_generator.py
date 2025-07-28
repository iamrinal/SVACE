import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import os

MODEL_NAME = "t5-base"
SAVE_DIR = "saved_generator"
BATCH_SIZE = 4
EPOCHS = 3
MAX_LENGTH = 256
LEARNING_RATE = 2e-5

# Load dataset
df = pd.read_csv("data/svace_dataset.csv")
df = df.dropna(subset=["code_with_bug", "flag", "fixed_code"])

# Prepare input and output
inputs = df.apply(lambda row: f"fix: [{row['flag']}] {row['code_with_bug']}", axis=1).tolist()
outputs = df['fixed_code'].tolist()

# Dataset class must be defined before use
class CodeFixDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        input_enc = tokenizer(self.inputs[idx], truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
        output_enc = tokenizer(self.outputs[idx], truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
        return {
            'input_ids': input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels': output_enc['input_ids'].squeeze()
        }

# Train-test split
train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42
)

# Force fresh download from HuggingFace
import transformers

tokenizer = transformers.T5Tokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir="./hf_cache",
    force_download=True
)

train_dataset = CodeFixDataset(train_inputs, train_outputs)
val_dataset = CodeFixDataset(val_inputs, val_outputs)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = transformers.T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    cache_dir="./hf_cache",
    force_download=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS * len(train_loader)
)

model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}")
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("✅ Generator training complete. Model saved to 'saved_generator/'") 