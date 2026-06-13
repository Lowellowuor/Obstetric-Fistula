import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import mlflow
import mlflow.pytorch
from src.data.database import DatabaseManager
from src.features.preprocessing import clean_text

class FistulaDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    rec_urgent = recall_score(labels, predictions, labels=[2], average=None)[0]  # urgent recall
    return {'accuracy': acc, 'urgent_recall': rec_urgent}

def train_distilbert(db_path: str, model_output_dir: str = "models_artifacts/distilbert_finetuned"):
    db = DatabaseManager(db_path)
    df = db.get_training_data(min_confidence=0.0)  # use ground truth labels
    # Clean texts
    df['clean_message'] = df['raw_message'].apply(clean_text)
    X = df['clean_message'].tolist()
    y = df['label'].tolist()
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    
    train_dataset = FistulaDataset(X_train, y_train, tokenizer)
    val_dataset = FistulaDataset(X_val, y_val, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="urgent_recall",
        greater_is_better=True,
        report_to="mlflow"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    with mlflow.start_run(run_name="distilbert_finetune"):
        mlflow.log_params({
            "model": "distilbert-base-uncased",
            "epochs": 3,
            "batch_size": 16,
            "max_length": 128
        })
        trainer.train()
        eval_results = trainer.evaluate()
        mlflow.log_metrics(eval_results)
        trainer.save_model(model_output_dir)
        mlflow.pytorch.log_model(model, "model")
    
    return trainer