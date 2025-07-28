import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import joblib
import os

MODEL_NAME = "microsoft/codebert-base"
SAVE_DIR = "saved_classifier"
LABEL_ENCODER_PATH = "label_encoder.joblib"
BATCH_SIZE = 8
EPOCHS = 3
MAX_LENGTH = 256
LEARNING_RATE = 2e-5

# Load dataset
df = pd.read_csv("data/svace_dataset.csv")
df = df.dropna(subset=["code_with_bug", "flag", "label"])

# Concatenate flag and code for input
inputs = df.apply(lambda row: f"[{row['flag']}] {row['code_with_bug']}", axis=1).tolist()

# Encode labels
label_encoder = LabelEncoder()
df['label_enc'] = label_encoder.fit_transform(df['label'])
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    inputs, df['label_enc'].tolist(), test_size=0.2, random_state=42
)

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

class SVACEDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

train_dataset = SVACEDataset(train_texts, train_labels)
val_dataset = SVACEDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
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
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
print("✅ Classifier training complete. Model saved to 'saved_classifier/'") 