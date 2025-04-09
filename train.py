import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
def save_checkpoint(model, optimizer, filepath="dr-detector.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Checkpoint saved at {filepath}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = {
                k: v.clone().detach() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early Stopping Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                return True
        return False
    

early_stopping = EarlyStopping(patience=15, min_delta=0, restore_best_weights=True)

def train(model,num_epochs, train_loader, val_loader):

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=6, min_lr=1e-5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # ------------------------- TRAINING -------------------------
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Lists to store predictions & labels for confusion matrix
        train_preds_list = []
        train_labels_list = []

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        if device.type == 'cuda':
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
        else:
            print("Running on CPU – no GPU memory info to display.")


        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            if isinstance(labels, (tuple, list)):
                labels = torch.tensor([int(lbl) for lbl in labels], dtype=torch.long)

            if isinstance(inputs, list):
                inputs = torch.stack(inputs)

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            # Collect preds for confusion matrix
            train_preds_list.append(predicted.detach().cpu().numpy())
            train_labels_list.append(labels.detach().cpu().numpy())

            train_accuracy = 100.0 * correct_train / total_train
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}",
                                      "Acc": f"{train_accuracy:.2f}%"})

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train

        # Flatten the lists for confusion matrix
        train_preds_list = np.concatenate(train_preds_list)
        train_labels_list = np.concatenate(train_labels_list)

        # ------------------------- VALIDATION -------------------------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_preds_list = []
        val_labels_list = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                if isinstance(labels, (tuple, list)):
                    labels = torch.tensor([int(lbl) for lbl in labels], dtype=torch.long)
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs)

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                # Collect preds for confusion matrix
                val_preds_list.append(predicted.detach().cpu().numpy())
                val_labels_list.append(labels.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val

        val_preds_list = np.concatenate(val_preds_list)
        val_labels_list = np.concatenate(val_labels_list)

        # ------------------- LOG & METRICS (Confusion Matrix) -------------------
        print(f"[Epoch {epoch+1}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_accuracy:.2f}%")

        # -- Training Confusion Matrix & Report
        train_cm = confusion_matrix(train_labels_list, train_preds_list)
        train_report = classification_report(train_labels_list, train_preds_list, digits=4)
        print("\nTraining Confusion Matrix:\n", train_cm)
        print("Training Classification Report:\n", train_report)

        # -- Validation Confusion Matrix & Report
        val_cm = confusion_matrix(val_labels_list, val_preds_list)
        val_report = classification_report(val_labels_list, val_preds_list, digits=4)
        print("Validation Confusion Matrix:\n", val_cm)
        print("Validation Classification Report:\n", val_report)

        # ------------------- Checkpoints & Scheduler -------------------
        save_checkpoint(model, optimizer, filepath="dr-detector.pth")
        scheduler.step(val_loss)
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")



        # Early Stopping
        if early_stopping(val_loss, model):
            print("Stopped early due to no improvement in val loss.")
            break
