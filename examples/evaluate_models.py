import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# Import the models
from models import Attention, LocalTrittention, MixedAttention, Trittention, TrittentionCube

# Define a simple dataset
def get_dataset():
    # Create some dummy data
    X = torch.randn(100, 10, 50)  # (num_samples, seq_length, feature_size)
    y = torch.randint(0, 2, (100, 10))  # Binary classification for each sequence step
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)

# Define evaluation metrics
def calculate_metrics(preds, labels):
    _, preds_max = torch.max(preds, 2)
    preds_max = preds_max.cpu().numpy()
    labels = labels.cpu().numpy()
    
    accuracy = (preds_max == labels).sum().item() / (labels.shape[0] * labels.shape[1])
    precision = precision_score(labels.flatten(), preds_max.flatten(), average='micro')
    recall = recall_score(labels.flatten(), preds_max.flatten(), average='micro')
    f1 = f1_score(labels.flatten(), preds_max.flatten(), average='micro')
    
    return accuracy, precision, recall, f1

# Function to evaluate a model
def evaluate_model(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        # Reshape outputs to (batch_size * seq_length, num_classes) for loss computation
        outputs = outputs.view(-1, outputs.size(-1))
        y_batch = y_batch.view(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        acc, prec, rec, f1 = calculate_metrics(outputs.view(X_batch.size(0), X_batch.size(1), -1), y_batch.view(X_batch.size(0), X_batch.size(1)))
        total_accuracy += acc
        total_precision += prec
        total_recall += rec
        total_f1 += f1
    
    num_batches = len(dataloader)
    return total_loss / num_batches, total_accuracy / num_batches, total_precision / num_batches, total_recall / num_batches, total_f1 / num_batches

# Main function to run the evaluation
def main():
    dataloader = get_dataset()
    criterion = nn.CrossEntropyLoss()
    
    # Instantiate models
    models = {
        'Attention': Attention(config),
        'LocalTrittention': LocalTrittention(config),
        'MixedAttention': MixedAttention(config),
        'Trittention': Trittention(config),
        'TrittentionCube': TrittentionCube(config)
    }
    
    # Store results
    results = {'Model': [], 'Loss': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'Time': []}

    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        start_time = time.time()
        loss, acc, prec, rec, f1 = evaluate_model(model, dataloader, criterion, optimizer)
        end_time = time.time()
        
        results['Model'].append(name)
        results['Loss'].append(loss)
        results['Accuracy'].append(acc)
        results['Precision'].append(prec)
        results['Recall'].append(rec)
        results['F1'].append(f1)
        results['Time'].append(end_time - start_time)
    
    # Save results to a file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'results_{timestamp}.csv')
    
    with open(results_file, 'w') as f:
        f.write("Model,Loss,Accuracy,Precision,Recall,F1,Time\n")
        for i in range(len(results['Model'])):
            f.write(f"{results['Model'][i]},{results['Loss'][i]},{results['Accuracy'][i]},{results['Precision'][i]},{results['Recall'][i]},{results['F1'][i]},{results['Time'][i]}\n")
    
    # Plot results
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.bar(results['Model'], results['Accuracy'], color='g', label='Accuracy')
    ax2.plot(results['Model'], results['Loss'], color='b', label='Loss')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Loss', color='b')
    
    fig.tight_layout()
    plt.title('Model Performance')
    plt.legend(loc='upper left')
    
    plot_file = os.path.join(results_dir, f'results_{timestamp}.png')
    plt.savefig(plot_file)
    plt.show()

if __name__ == '__main__':
    # Define a sample config
    class Config:
        hidden_size = 50
        num_attention_heads = 5
        attention_probs_dropout_prob = 0.1

    config = Config()
    main()
