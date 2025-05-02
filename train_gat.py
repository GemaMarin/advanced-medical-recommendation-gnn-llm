import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict
import numpy as np
import math

# ---------------------- Modelo GAT --------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNN_GAT(nn.Module):
    def __init__(self, hidden_channels, dropout=0.3, num_layers=3, heads=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Capa 1
        self.convs.append(
            GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, add_self_loops=False)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, add_self_loops=False)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Última capa
        self.convs.append(
            GATConv(hidden_channels, 128, heads=heads, concat=False, add_self_loops=False)
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class ClassifierMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
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
    def __init__(self, hidden_channels, dropout=0.3):
        super().__init__()
        self.drug_lin = nn.Linear(768, hidden_channels)
        self.disorder_lin = nn.Linear(768, hidden_channels)
        self.protein_lin = nn.Linear(768, hidden_channels)
        self.gene_lin = nn.Linear(768, hidden_channels)
        self.gnn = GNN_GAT(hidden_channels, dropout=dropout, num_layers=3)
        self.hetero_gnn = None
        self.classifier = ClassifierMLP(in_channels=128, hidden_channels=128)

    def forward(self, data: HeteroData):
        x_dict = {
            "Drug": self.drug_lin(data["Drug"].x),
            "Disorder": self.disorder_lin(data["Disorder"].x),
            "Protein": self.protein_lin(data["Protein"].x),
            "Gene": self.gene_lin(data["Gene"].x),
        }
        if self.hetero_gnn is None:
            node_types, edge_types = data.metadata()
            self.hetero_gnn = to_hetero(self.gnn, (node_types, edge_types))
            self.hetero_gnn = self.hetero_gnn.to(data["Drug"].x.device)
        x_dict = self.hetero_gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["Drug"],
            x_dict["Disorder"],
            data["Drug", "DrugHasIndication", "Disorder"].edge_label_index,
        )
        return pred

# ---------------------- Métricas --------------------------
def compute_local_mrr(scores_labels):
    sorted_ = sorted(scores_labels, key=lambda x: x[0], reverse=True)
    for i, (_, label) in enumerate(sorted_):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0

def compute_local_ndcg(scores_labels, ndcg_k=None):
    sorted_ = sorted(scores_labels, key=lambda x: x[0], reverse=True)
    n = len(sorted_)
    if ndcg_k is None or ndcg_k > n:
        ndcg_k = n
    dcg = sum((2**rel - 1) / math.log2(i + 2) for i, (_, rel) in enumerate(sorted_[:ndcg_k]))
    sorted_ideal = sorted(scores_labels, key=lambda x: x[1], reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, (_, rel) in enumerate(sorted_ideal[:ndcg_k]))
    return dcg / idcg if idcg > 0 else 0.0

@torch.no_grad()
def evaluate_local_ranking_metrics(model, data, device='cpu', ndcg_k=None):
    model.eval()
    data = data.to(device)
    pred = model(data)
    scores = torch.sigmoid(pred).cpu().numpy()
    labels = data["Drug", "DrugHasIndication", "Disorder"].edge_label.cpu().numpy()
    edge_index = data["Drug", "DrugHasIndication", "Disorder"].edge_label_index.cpu().numpy()
    drug_ids = edge_index[0]
    groups = defaultdict(list)
    for i in range(len(scores)):
        groups[drug_ids[i]].append((scores[i], labels[i]))
    mrr_list = []
    ndcg_list = []
    for slist in groups.values():
        if any(x[1] == 1 for x in slist):
            mrr_list.append(compute_local_mrr(slist))
            ndcg_list.append(compute_local_ndcg(slist, ndcg_k))
    return np.mean(mrr_list), np.mean(ndcg_list)

def compute_recall_at_k(probs, labels, k=50):
    sorted_indices = np.argsort(-probs)
    top_k_indices = sorted_indices[:k]
    positives_in_topk = labels[top_k_indices].sum()
    return positives_in_topk / (labels.sum() + 1e-8)

# ---------------- Entrenamiento BCE -----------------
def train_epoch_bce(model, data, optimizer, device='cpu', calc_recall_topk=None):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    pred = model(data)
    labels = data["Drug", "DrugHasIndication", "Disorder"].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, labels.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()
    probs = torch.sigmoid(pred).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0
    acc = accuracy_score(labels_np, (probs >= 0.5))
    recall_at_k = compute_recall_at_k(probs, labels_np, k=calc_recall_topk) if calc_recall_topk else None
    return loss.item(), auc, acc, recall_at_k

@torch.no_grad()
def evaluate_bce(model, data, device='cpu', calc_recall_topk=None):
    model.eval()
    data = data.to(device)
    pred = model(data)
    labels = data["Drug", "DrugHasIndication", "Disorder"].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, labels.float())
    probs = torch.sigmoid(pred).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0
    acc = accuracy_score(labels_np, (probs >= 0.5))
    recall_at_k = compute_recall_at_k(probs, labels_np, k=calc_recall_topk) if calc_recall_topk else None
    return loss.item(), auc, acc, recall_at_k

def train_loop_bce(model, train_data, val_data, epochs=200, lr=1e-4, weight_decay=1e-4,
                   model_file="gnn_model_gat_bce.pth", device='cpu',
                   calc_recall_topk=100, calc_local_ranking=True, ndcg_k=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    best_val_auc = 0.0
    patience = 50
    early_stop_counter = 0
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss, train_auc, train_acc, train_recall = train_epoch_bce(
            model, train_data, optimizer, device, calc_recall_topk)
        val_loss, val_auc, val_acc, val_recall = evaluate_bce(
            model, val_data, device, calc_recall_topk)
        val_mrr_local, val_ndcg_local = 0.0, 0.0
        if calc_local_ranking:
            val_mrr_local, val_ndcg_local = evaluate_local_ranking_metrics(
                model, val_data, device, ndcg_k)
        scheduler.step(val_auc)
        print(f"Epoch {epoch:02d} | TrainLoss={train_loss:.4f} | AUC={train_auc:.4f} | "
              f"ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f} | MRR_local={val_mrr_local:.4f} | "
              f"NDCG@{ndcg_k}={val_ndcg_local:.4f}")
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