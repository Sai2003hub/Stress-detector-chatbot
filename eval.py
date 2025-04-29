# eval.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = "C:/Users/WinX/Downloads/Stress"
LIAR_TEST_PATH = os.path.join(BASE_DIR, "test.tsv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_lie_model")

# Binary label mapping
binary_label_mapping = {
    'true': 1,
    'mostly-true': 1,
    'half-true': 1,
    'barely-true': 0,
    'false': 0,
    'pants-fire': 0
}

# Load the LIAR test dataset
def load_liar_test_dataset():
    df = pd.read_csv(LIAR_TEST_PATH, sep='\t', header=None)
    statements = df[2].tolist()
    labels = df[1].map(binary_label_mapping).tolist()
    return [{"statement": statement, "label": label} for statement, label in zip(statements, labels)]

# Create a custom dataset class
class LiarDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.statements = [item["statement"] for item in data]
        self.labels = [item["label"] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        statement = str(self.statements[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            statement,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Evaluate the model
def evaluate_model():
    # Load the test dataset
    test_data = load_liar_test_dataset()

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # Create test dataset
    test_dataset = LiarDataset(test_data, tokenizer)

    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Evaluate the model
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            input_ids = item['input_ids'].unsqueeze(0)
            attention_mask = item['attention_mask'].unsqueeze(0)
            label = item['labels'].item()

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

            all_preds.append(pred)
            all_labels.append(label)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy on test set: {accuracy:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Deceptive', 'Truthful'], yticklabels=['Deceptive', 'Truthful'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('C:/Users/WinX/Downloads/Stress/confusion_matrix.png')
    print("Confusion matrix saved to C:/Users/WinX/Downloads/Stress/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()