# finetune_liar_binary.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Define paths
BASE_DIR = "C:/Users/WinX/Downloads/Stress"
LIAR_TRAIN_PATH = os.path.join(BASE_DIR, "train.tsv")
LIAR_VALID_PATH = os.path.join(BASE_DIR, "valid.tsv")
LIAR_TEST_PATH = os.path.join(BASE_DIR, "test.tsv")

# Binary label mapping
binary_label_mapping = {
    'true': 1,
    'mostly-true': 1,
    'half-true': 1,
    'barely-true': 0,
    'false': 0,
    'pants-fire': 0
}

# Load the LIAR dataset
def load_liar_dataset():
    # Load train, validation, and test sets
    train_df = pd.read_csv(LIAR_TRAIN_PATH, sep='\t', header=None)
    valid_df = pd.read_csv(LIAR_VALID_PATH, sep='\t', header=None)
    test_df = pd.read_csv(LIAR_TEST_PATH, sep='\t', header=None)

    # Combine train and valid for training
    df = pd.concat([train_df, valid_df])
    statements = df[2].tolist()
    labels = df[1].map(binary_label_mapping).tolist()

    # Test set for evaluation
    test_statements = test_df[2].tolist()
    test_labels = test_df[1].map(binary_label_mapping).tolist()

    train_data = [{"statement": statement, "label": label} for statement, label in zip(statements, labels)]
    test_data = [{"statement": statement, "label": label} for statement, label in zip(test_statements, test_labels)]
    return train_data, test_data

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

# Main fine-tuning function
def main():
    # Load the dataset
    train_data, test_data = load_liar_dataset()

    # Split train_data into train and validation sets
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Create datasets
    train_dataset = LiarDataset(train_data, tokenizer)
    val_dataset = LiarDataset(val_data, tokenizer)
    test_dataset = LiarDataset(test_data, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./finetune_results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Define compute_metrics function for evaluation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

    # Save the model and tokenizer
    model.save_pretrained('C:/Users/WinX/Downloads/Stress/saved_lie_model')
    tokenizer.save_pretrained('C:/Users/WinX/Downloads/Stress/saved_lie_model')
    print("Fine-tuned model saved to C:/Users/WinX/Downloads/Stress/saved_lie_model")

if __name__ == "__main__":
    main()