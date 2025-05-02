import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from transformers import BertModel

# ----------- Modelo con Fine-tuning BioBERT y GraphSAGE -----------
class GNN(nn.Module):
    def __init__(self, hidden_channels=768, dropout=0.3, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(self.bns[i](conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class ClassifierMLP(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=256):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, x_drug, x_disorder, edge_label_index):
        edge_feat_drug = x_drug[edge_label_index[0]]
        edge_feat_disorder = x_disorder[edge_label_index[1]]
        x = torch.cat([edge_feat_drug, edge_feat_disorder], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return x


class Model(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bio_bert = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
        for param in self.bio_bert.parameters():
            param.requires_grad = True

        self.gnn = GNN(hidden_channels=768, dropout=dropout)
        self.hetero_gnn = None
        self.classifier = ClassifierMLP(in_channels=768, hidden_channels=256)

    def forward(self, data):
        x_dict = {}
        device = next(self.parameters()).device  # <-- Cambia aquí para usar el dispositivo del modelo

        for node_type in ["Drug", "Disorder", "Protein", "Gene"]:
            input_ids = data[node_type].input_ids.to(device)
            attention_mask = data[node_type].attention_mask.to(device)
            x = self.bio_bert(input_ids=input_ids, attention_mask=attention_mask)[1]
            x_dict[node_type] = x

        if self.hetero_gnn is None:
            node_types, edge_types = data.metadata()
            self.hetero_gnn = to_hetero(self.gnn, (node_types, edge_types)).to(device)

        x_dict = self.hetero_gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["Drug"],
            x_dict["Disorder"],
            data["Drug", "DrugHasIndication", "Disorder"].edge_label_index.to(device),
        )
        return pred

def train_epoch_bce(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_examples = 0
    all_labels = []
    all_preds = []

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        pred = model(batch)
        labels = batch["Drug", "DrugHasIndication", "Disorder"].edge_label.float().to(device)
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
        
        all_labels.append(labels.detach().cpu())
        all_preds.append(torch.sigmoid(pred).detach().cpu())

    avg_loss = total_loss / total_examples
    auc = roc_auc_score(torch.cat(all_labels), torch.cat(all_preds))

    return avg_loss, auc

@torch.no_grad()
def evaluate_bce(model, loader, device):
    model.eval()
    total_loss = 0
    total_examples = 0
    all_labels = []
    all_preds = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        labels = batch["Drug", "DrugHasIndication", "Disorder"].edge_label.float().to(device)
        loss = F.binary_cross_entropy_with_logits(pred, labels)

        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
        
        all_labels.append(labels.cpu())
        all_preds.append(torch.sigmoid(pred).cpu())

    avg_loss = total_loss / total_examples
    auc = roc_auc_score(torch.cat(all_labels), torch.cat(all_preds))

    return avg_loss, auc


# ------- Entrenamiento optimizado con Fine-tuning -------
def train_loop_bce(model, train_loader, val_loader, epochs=150, lr=5e-5, weight_decay=1e-5,
                   model_file="gnn_model_finetuned.pth", device='cuda', patience=15):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_val_auc = 0.0
    early_stop_counter = 0
    model = model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_epoch_bce(model, train_loader, optimizer, device)
        val_loss, val_auc = evaluate_bce(model, val_loader, device)
        scheduler.step(val_auc)

        print(f"Epoch {epoch:02d} | TrainLoss={train_loss:.4f} | TrainAUC={train_auc:.4f} | "
              f"ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_file)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(model_file))
    return model
