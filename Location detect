import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class TemporalAwareBlock(nn.Module):
    """Temporal Aware Block"""
    def __init__(self, in_dim, expansion=4):
        super().__init__()
        hidden_dim = in_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)
        )
        self.temporal_attn = nn.MultiheadAttention(in_dim, 4, batch_first=True)

    def forward(self, x):
        residual = x
        attn_out, _ = self.temporal_attn(x, x, x)
        transformed = self.net(attn_out)
        return transformed + residual


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.proj1 = nn.Linear(dim1, 512)
        self.proj2 = nn.Linear(dim2, 512)
        self.query = nn.Linear(512, 512)
        self.key = nn.Linear(512, 512)
        self.value = nn.Linear(512, 512)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        Q1 = self.query(x1)
        K2 = self.key(x2)
        V2 = self.value(x2)
        attn_matrix = torch.matmul(Q1, K2.transpose(-2, -1)) / np.sqrt(512)
        attn_weights = torch.softmax(attn_matrix, dim=-1)
        fused1 = torch.matmul(attn_weights, V2)
        Q2 = self.query(x2)
        K1 = self.key(x1)
        V1 = self.value(x1)
        attn_matrix = torch.matmul(Q2, K1.transpose(-2, -1)) / np.sqrt(512)
        attn_weights = torch.softmax(attn_matrix, dim=-1)
        fused2 = torch.matmul(attn_weights, V1)
        return self.gamma * (fused1 + fused2) + x1 + x2


class OptimizedFDIA(nn.Module):
    def __init__(self, input_dim=34, num_classes=34, fusion_type='weighted'):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.Linear(input_dim, 256),
            TemporalAwareBlock(256),
            TemporalAwareBlock(256),
            nn.LayerNorm(256)
        )
        self.conv_branch = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            DepthwiseSeparableConv1d(64, 128, 3),
            SqueezeExcitation(128),
            nn.Conv1d(128, 256, 3, padding=1),
            DepthwiseSeparableConv1d(256, 512, 3),
            SqueezeExcitation(512),
            nn.Flatten(),
            nn.Linear(34*512, 512)
        )
        if fusion_type == 'concat':
            self.fusion = ConcatenationFusion(dim1=256, dim2=512)
        elif fusion_type == 'weighted':
            self.fusion = LearnableWeightedFusion(dim1=256, dim2=512)
        else:
            raise ValueError("Choose 'concat' or 'weighted' fusion method")
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        global_feat = self.global_branch(x)
        conv_feat = self.conv_branch(x.unsqueeze(1))  
        fused = self.fusion(global_feat, conv_feat)
        return self.classifier(fused)


class LearnableWeightedFusion(nn.Module):
    """Learnable Weighted Fusion"""
    def __init__(self, dim1, dim2):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj1 = nn.Linear(dim1, 512) if dim1 != 512 else nn.Identity()
        self.proj2 = nn.Linear(dim2, 512) if dim2 != 512 else nn.Identity()

    def forward(self, x1, x2):
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        return self.alpha * x1 + (1 - self.alpha) * x2


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_weights = nn.Parameter(torch.ones(34))

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * BCE_loss)
        weighted_loss = focal_loss * self.label_weights.unsqueeze(0)
        return weighted_loss.mean()


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_split_data():
    """Load and split data"""
    X = pd.read_csv(r'0.1-X0.08.csv').values
    y = pd.read_csv(r'0.1-Y0.08.csv').values
    damaged_idx = np.where(y.sum(axis=1) > 0)[0]
    normal_idx = np.where(y.sum(axis=1) == 0)[0]
    dmg_X_train, dmg_X_temp, dmg_y_train, dmg_y_temp = train_test_split(X[damaged_idx], y[damaged_idx], test_size=0.4, random_state=42)
    dmg_X_val, dmg_X_test, dmg_y_val, dmg_y_test = train_test_split(dmg_X_temp, dmg_y_temp, test_size=0.5, random_state=42)
    nrm_X_train, nrm_X_temp, nrm_y_train, nrm_y_temp = train_test_split(X[normal_idx], y[normal_idx], test_size=0.4, random_state=42)
    nrm_X_val, nrm_X_test, nrm_y_val, nrm_y_test = train_test_split(nrm_X_temp, nrm_y_temp, test_size=0.5, random_state=42)
    train_dmg = len(dmg_X_train)
    train_nrm = int(train_dmg * 7 / 3)
    X_train = np.concatenate([dmg_X_train, nrm_X_train[:train_nrm]], axis=0)
    y_train = np.concatenate([dmg_y_train, nrm_y_train[:train_nrm]], axis=0)
    test_samples = min(len(dmg_X_test), len(nrm_X_test))
    X_test = np.concatenate([dmg_X_test[:test_samples], nrm_X_test[:test_samples]], axis=0)
    y_test = np.concatenate([dmg_y_test[:test_samples], nrm_y_test[:test_samples]], axis=0)
    X_val = np.concatenate([dmg_X_val, nrm_X_val], axis=0)
    y_val = np.concatenate([dmg_y_val, nrm_y_val], axis=0)

    def shuffle_data(X, y):
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    return shuffle_data(X_train, y_train) + shuffle_data(X_val, y_val) + shuffle_data(X_test, y_test)


def print_dataset_stats(y_train, y_val, y_test):
    """Print dataset statistics"""
    def counter(y):
        return sum(y.sum(1) > 0), sum(y.sum(1) == 0)
    
    print("\nDataset Distribution:")
    print(f"Train Set: Normal samples={counter(y_train)[1]}, Damaged samples={counter(y_train)[0]}")
    print(f"Validation Set: Normal samples={counter(y_val)[1]}, Damaged samples={counter(y_val)[0]}")
    print(f"Test Set: Normal samples={counter(y_test)[1]}, Damaged samples={counter(y_test)[0]}")


def evaluate(model, loader, device):
    """Model evaluation"""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(targets.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs_flat = all_probs.ravel()
    all_labels_flat = all_labels.ravel()
    fpr, tpr, thresholds = roc_curve(all_labels_flat, all_probs_flat)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels_flat, all_probs_flat)
    pr_auc = auc(recall, precision)
    roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    roc_data.to_csv('roc_data.csv', index=False)
    preds = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_labels.flatten(), preds.flatten())
    precision_score_val = precision_score(all_labels, preds, average='weighted', zero_division=0)
    recall_score_val = recall_score(all_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)
    racc = np.mean([np.all(l == p) for l, p in zip(all_labels, preds)])
    fpr_list = []
    for i in range(all_labels.shape[1]):
        tn, fp, fn, tp = confusion_matrix(all_labels[:, i], preds[:, i], labels=[0, 1]).ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_list.append(fpr_val)
    avg_fpr = np.mean(fpr_list)
    return {
        'accuracy': accuracy,
        'precision': precision_score_val,
        'recall': recall_score_val,
        'f1': f1,
        'racc': racc,
        'fpr': avg_fpr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    print_dataset_stats(y_train, y_val, y_test)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print(f"\nData dimension check: X_train.shape={X_train.shape}")
    train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=128)
    test_loader = DataLoader(CustomDataset(X_test, y_test), batch_size=128)
    model = OptimizedFDIA(X_train.shape[1], y_train.shape[1], fusion_type='weighted').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=100, pct_start=0.2)
    criterion = AdaptiveFocalLoss(alpha=0.5, gamma=2).to(device)
    test_input = torch.randn(3, 34).to(device)
    with torch.no_grad():
        output = model(test_input)
        print(f"\nDimension test result: Input shape {test_input.shape} → Output shape {output.shape}")
        assert output.shape == (3, 34), "Model output dimension error!"

    best_f1 = 0
    patience_counter = 0
    print("\nStarting training...")
    print("Epoch | Train Loss | "
          "Train Acc | Train Prec | Train Rec | Train F1 | Train Racc | Train FPR | Train ROC-AUC | Train PR-AUC | "
          "Val Acc | Val Prec | Val Rec | Val F1 | Val Racc | Val FPR | Val ROC-AUC | Val PR-AUC | "
          "Test Acc | Test Prec | Test Rec | Test F1 | Test Racc | Test FPR | Test ROC-AUC | Test PR-AUC")
    print("-" * 150)

    for epoch in range(100):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        scheduler.step(val_metrics['f1'])

        print(f"{epoch + 1:5d} | {total_loss / len(train_loader):.4f} | "
              f"{train_metrics['accuracy']:.4f} | {train_metrics['precision']:.4f} | {train_metrics['recall']:.4f} | "
              f"{train_metrics['f1']:.4f} | {train_metrics['racc']:.4f} | {train_metrics['fpr']:.4f} | "
              f"{train_metrics['roc_auc']:.4f} | {train_metrics['pr_auc']:.4f} | "
              f"{val_metrics['accuracy']:.4f} | {val_metrics['precision']:.4f} | {val_metrics['recall']:.4f} | "
              f"{val_metrics['f1']:.4f} | {val_metrics['racc']:.4f} | {val_metrics['fpr']:.4f} | "
              f"{val_metrics['roc_auc']:.4f} | {val_metrics['pr_auc']:.4f} | "
              f"{test_metrics['accuracy']:.4f} | {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | "
              f"{test_metrics['f1']:.4f} | {test_metrics['racc']:.4f} | {test_metrics['fpr']:.4f} | "
              f"{test_metrics['roc_auc']:.4f} | {test_metrics['pr_auc']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    final_metrics = evaluate(model, test_loader, device)
    print("\nFinal test results:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"FPR: {final_metrics['fpr']:.4f}")
    print(f"AUC-ROC: {final_metrics['roc_auc']:.4f}")
    print(f"AUC-PR: {final_metrics['pr_auc']:.4f}")
    print("ROC curve data saved to 'roc_data.csv', can be plotted in Origin.")

if __name__ == "__main__":
    main() 
