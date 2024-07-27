import streamlit as st
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import evaluate
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Streamlit title
st.title("Sentiment Analysis with HuggingFace")

# Load dataset
emotions = load_dataset("dair-ai/emotion", "split")

# Display dataset info
st.write(emotions)

train_ds = emotions["train"]
st.write(train_ds)
st.write(f"Number of training samples: {len(train_ds)}")

# Display first sample
st.write(train_ds[1])

# Display column names and features
st.write(train_ds.column_names)
st.write(train_ds.features)

# Display first 5 samples
st.write(train_ds[:5])
st.write(train_ds["text"][:5])

# Convert to pandas DataFrame
emotions.set_format(type="pandas")
df = emotions["train"][:]
st.write(df.head())

# Function to convert label integers to strings
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)
st.write(df.head())

# Plot frequency of classes
st.write("Frequency of Classes")
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
st.pyplot(plt)

# Plot words per tweet
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
st.pyplot(plt)

# Reset format
emotions.reset_format()

# Tokenization example
text = "It is fun to work with NLP using HuggingFace."
tokenized_text = list(text)
st.write(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
st.write(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
st.write(input_ids)

# One-hot encoding
input_ids_tensor = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids_tensor, num_classes=len(token2idx))
st.write(f"Token: {tokenized_text[0]}")
st.write(f"Tensor index: {input_ids[0]}")
st.write(f"One-hot: {one_hot_encodings[0]}")

# Tokenization with transformers
tokenized_text = text.split()
st.write(tokenized_text)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
distbert_tokenize = DistilBertTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)
st.write(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
st.write(tokens)

st.write(tokenizer.convert_tokens_to_string(tokens))
st.write(f"Vocab size: {tokenizer.vocab_size}")
st.write(f"Model max length: {tokenizer.model_max_length}")

# Tokenize function
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Tokenize dataset
st.write(tokenize(emotions["train"][:2]))
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Display dataset columns
st.write(emotions_encoded["train"].column_names)

# Load model
num_labels = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

# Evaluation
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="distilbert-emotion",
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to="none"
)

# Data loaders
train_loader = DataLoader(emotions_encoded["train"], batch_size=128, shuffle=True, num_workers=4)
eval_loader = DataLoader(emotions_encoded["validation"], batch_size=128, shuffle=False, num_workers=4)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_loader,
    eval_dataset=eval_loader,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Predictions
preds_output = trainer.predict(emotions_encoded["validation"])
st.write(preds_output.metrics)

# Confusion matrix
y_preds = np.argmax(preds_output.predictions, axis=1)
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Display confusion matrix
plot_confusion_matrix(y_preds, emotions_encoded["validation"]["label"], emotions["train"].features["label"].names)