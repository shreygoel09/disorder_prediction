import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# Hyperparameters dictionary
path = "/workspace/a03-sgoel/FusOn-pLM"
hyperparams = {
    "max_length": 4405,
    "train_data": path + "/data/CAID_Training_Dataset_july2024.csv",
    "test_data": path + "/data/CAID_Testing_Dataset_july2024.csv",
    'esm_model_path': "facebook/esm2_t33_650M_UR50D",
    'fuson_model_path': "ChatterjeeLab/FusOn-pLM",
    "batch_size": 1,
    "learning_rate": 5e-5,
    "num_epochs": 2,
    "num_layers": 4,
    "num_heads": 8,
    "dropout": 0.5
}


# Helper functions to obtain all embeddings for a sequences
def load_models(esm_model_path, fuson_model_path):
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_path)
    esm_model = AutoModel.from_pretrained(esm_model_path)
    fuson_tokenizer = AutoTokenizer.from_pretrained(fuson_model_path)
    fuson_model = AutoModel.from_pretrained(fuson_model_path)
    return esm_tokenizer, esm_model, fuson_tokenizer, fuson_model

def get_fuson_embeds(sequence, model, tokenizer):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=4405)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)[1:-1, :]
    embeddings = embeddings.cpu().numpy()
    return embeddings

def get_esm_embeds(sequence, model, tokenizer):
    inputs = tokenizer(sequence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings

def get_latents(embedding_type, esm_model_path, fuson_model_path, sequence):
    esm_tokenizer, esm_model, fuson_tokenizer, fuson_model = load_models(esm_model_path, fuson_model_path)
    if embedding_type == "esm":
        embeddings = get_esm_embeds(sequence, esm_model, esm_tokenizer)
    elif embedding_type == "fuson":
        embeddings = get_fuson_embeds(sequence, fuson_model, fuson_tokenizer)
    return embeddings


# Dataset class that loads embeddings and labels
class DisorderDataset(Dataset):
    def __init__(self, embedding_type, csv_file, esm_model_path, fuson_model_path):
        super(DisorderDataset, self).__init__()
        self.data = pd.read_csv(csv_file).head(5)
        self.embedding_type = embedding_type
        self.esm_model_path = esm_model_path
        self.fuson_model_path = fuson_model_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['Sequence']
        embeddings = get_latents(self.embedding_type, self.esm_model_path, self.fuson_model_path, sequence)

        # Convert string representations of labels to floats
        label_str = self.data.iloc[idx]['Label']
        label_str = label_str[1:-1] 
        labels = [int(num) for num in label_str.split(',')]
        labels = torch.tensor(labels, dtype=torch.float) 

        # Check if lenght of labels list and embedding's sequence length dimension are the same
        if len(labels) != embeddings.shape[0]:
            embeddings = torch.tensor(embeddings, dtype=torch.float32)[1:-1]  # if not same then bos/eos tokens are present so remove them
        else:
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        return embeddings, labels

# Transformer model class
class DisorderPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
        super(DisorderPredictor, self).__init__()
        self.embedding_dim = input_dim
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings):
        attn_out, _ = self.self_attention(embeddings, embeddings, embeddings) # Start with embeddings as random Q, K, V vectors
        transformer_out = self.transformer_encoder(attn_out) # Get outputs from encoder layers
        logits = self.classifier(transformer_out) # Linear classifier
        probs = self.softmax(logits.squeeze(-1)) # Softmax for probabilities
        return probs  # Get probabilities of dimension seq_len
    

# Training function
def train(model, train_loader, optimizer, criterion, device):
    """
    Trains the model for a single epoch.
    Args:
        model (nn.Module): model that will be trained
        dataloader (DataLoader): PyTorch DataLoader with training data
        optimizer (torch.optim): optimizer
        criterion (nn.Module): loss function
        device (torch.device): device (GPU or CPU to train the model
    Returns:
        total_loss (float): model loss
    """
    # Training loop
    model.train()
    train_loss = 0
    total_steps = len(train_loader)
    update_interval = total_steps // 4

    prog_bar = tqdm(total=total_steps, leave=True, file=sys.stdout)
    for step, batch in enumerate(train_loader, start=1):
        embeddings, labels = batch
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if step % update_interval == 0 or step == total_steps:
            prog_bar.update(update_interval)
            sys.stdout.flush()
    prog_bar.close()

    return train_loss/len(train_loader)


# Evaluation function
def evaluate(model, dataloader, device):
    """
    Performs inference on a trained model
    Args:
        model (nn.Module): the trained model
        dataloader (DataLoader): PyTorch DataLoader with testing data
        device (torch.device): device (GPU or CPU) to be used for inference
    Returns:
        preds (list): predicted per-residue disorder labels
        true_labels (list): ground truth per-residue disorder labels
    """
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            preds.append(outputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    return preds, true_labels

# Metrics calculation
def calculate_metrics(preds, labels, threshold=0.5):
    """
    Calculates metrics to assess model performance
    Args:
        preds (list): model's predictions
        labels (list): ground truth labels
        threshold (float): minimum threshold a prediction must be met to be considered disordered
    Returns:
        accuracy (float): accuracy
        precision (float): precision
        recall (float): recall
        f1 (float): F1 score
        roc_auc (float): AUROC score
    """
    flat_binary_preds, flat_prob_preds, flat_labels = [], [], []

    for pred, label in zip(preds, labels):
        flat_binary_preds.extend((pred > threshold).astype(int).flatten())
        flat_prob_preds.extend(pred.flatten())
        flat_labels.extend(label.flatten())

    flat_binary_preds = np.array(flat_binary_preds)
    flat_prob_preds = np.array(flat_prob_preds)
    flat_labels = np.array(flat_labels)

    accuracy = accuracy_score(flat_labels, flat_binary_preds)
    precision = precision_score(flat_labels, flat_binary_preds)
    recall = recall_score(flat_labels, flat_binary_preds)
    f1 = f1_score(flat_labels, flat_binary_preds)
    roc_auc = roc_auc_score(flat_labels, flat_prob_preds)

    return accuracy, precision, recall, f1, roc_auc


if __name__ == "__main__":
    best_val_loss = float('inf')
    best_model = None

    for embedding_type in ['esm', 'fuson']:
        # Load train and test dataset
        train_dataset = DisorderDataset(embedding_type=embedding_type,
                                        csv_file=hyperparams['train_data'],
                                        esm_model_path=hyperparams['esm_model_path'],
                                        fuson_model_path=hyperparams['fuson_model_path'])
        test_dataset = DisorderDataset(embedding_type=embedding_type,
                                       csv_file=hyperparams['test_data'],
                                       esm_model_path=hyperparams['esm_model_path'],
                                       fuson_model_path=hyperparams['fuson_model_path'])
        
        # Load datasets into DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

        # Set device to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### Grid search to explore hyperparameter space
        # Define hyperparameters
        param_grid = {
            'learning_rate': [5e-5],
            'batch_size': [1],
            'num_heads': [5, 10],
            'num_layers': [2],
            'dropout': [0.5],
            'num_epochs': [2]
        }

        # Loop over the parameter grid
        grid = ParameterGrid(param_grid)
        for params in grid:
            # Update hyperparameters
            hyperparams.update(params)
            
            # Update model with the new set of hyperparms
            input_dim, hidden_dim = 1280, 1280
            model = DisorderPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=hyperparams["num_layers"],
                num_heads=hyperparams["num_heads"],
                dropout=hyperparams['dropout']
            )
            model = model.to(device) # Push model to GPU
            
            # Update optimizer
            optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
            criterion = nn.BCELoss()
            num_epochs = hyperparams['num_epochs']

            # Train 
            for epoch in range(hyperparams["num_epochs"]):
                print(f"EPOCH {epoch+1}/{hyperparams['num_epochs']}")
                train_loss = train(model, train_dataloader, optimizer, criterion, device)
                print(f"TRAIN LOSS: {train_loss:.4f}")
                sys.stdout.flush()

            # Evaluate model on test sequences
            print("TEST METRICS:")
            test_preds, test_labels = evaluate(model, test_dataloader, device)
            test_metrics = calculate_metrics(test_preds, test_labels)
            print(f"Accuracy: {test_metrics[0]:.4f}")
            print(f"Precision: {test_metrics[1]:.4f}")
            print(f"Recall: {test_metrics[2]:.4f}")
            print(f"F1 Score: {test_metrics[3]:.4f}")
            print(f"ROC AUC: {test_metrics[4]:.4f}")
            print(f"\n")
            sys.stdout.flush()

            ### Save model and metrics for this hyperparameter combination
            folder_name = f"{path}/trained_models/{embedding_type}/lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}_hd{hidden_dim}_epochs{hyperparams['num_epochs']}_layers{hyperparams['num_layers']}_heads{hyperparams['num_heads']}_drpt{hyperparams['dropout']}"
            os.makedirs(folder_name, exist_ok=True)

            # Save current model for this hyperparameter combination
            model_file_path = os.path.join(folder_name, "model.pth")
            torch.save(model.state_dict(), model_file_path)

            # Save hyperparameters and test metrics to txt file
            output_file_path = os.path.join(folder_name, "hyperparams_and_test_results.txt")
            with open(output_file_path, 'w') as out_file:
                for key, value in hyperparams.items():
                    out_file.write(f"{key}: {value}\n")
                
                out_file.write("\nTEST METRICS:\n")
                out_file.write(f"Accuracy: {test_metrics[0]:.4f}\n")
                out_file.write(f"Precision: {test_metrics[1]:.4f}\n")
                out_file.write(f"Recall: {test_metrics[2]:.4f}\n")
                out_file.write(f"F1 Score: {test_metrics[3]:.4f}\n")
                out_file.write(f"ROC AUC: {test_metrics[4]:.4f}\n")