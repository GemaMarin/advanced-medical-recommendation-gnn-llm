import pandas as pd
import numpy as np
import pickle
import torch
from torch_geometric.data import HeteroData
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import time
import matplotlib.pyplot as plt
import random
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from torch import Tensor
from sklearn.metrics import roc_auc_score, accuracy_score
import tqdm
import os
from sklearn.metrics import roc_auc_score, accuracy_score
import copy
import pandas as pd
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
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
        for node_type in ["Drug", "Disorder", "Protein", "Gene"]:
            # Mover los tensores al dispositivo
            input_ids = data[node_type].input_ids.to(device)
            attention_mask = data[node_type].attention_mask.to(device)
            
            # Pasamos los input_ids y attention_mask a BioBERT
            x = self.bio_bert(input_ids=input_ids, attention_mask=attention_mask)[1]
            data[node_type].x = x  # Asignamos las características a `x` en los nodos
            x_dict[node_type] = x  # Guardamos en el diccionario para usarlo más tarde

        if self.hetero_gnn is None:
            node_types, edge_types = data.metadata()
            self.hetero_gnn = to_hetero(self.gnn, (node_types, edge_types)).to(device)

        x_dict = self.hetero_gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["Drug"],
            x_dict["Disorder"],
            data["Drug", "DrugHasIndication", "Disorder"].edge_label_index,
        )
        return pred



#CARGAR DATOS 
# Rutas
nodes_csv_path = 'nodes.csv'
edges_csv_path = 'edges.csv'

# Cargar datos
nodes_df = pd.read_csv(nodes_csv_path, low_memory=False)
edges_df = pd.read_csv(edges_csv_path, low_memory=False)

# Inicializa Tokenizer BioBERT
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

#Paso 2: Generar textos para embeddings
# Función para generar textos por nodo
def generate_node_text(row):
    node_type = row['type']
    
    if node_type in ['Gene', 'Protein']:
        parts = [row['displayName'], row['description'], row['symbols']]
    elif node_type == 'Drug':
        parts = [row['displayName'], row['description'], row['drugCategories'], row['synonyms']]
    elif node_type == 'Disorder':
        parts = [row['displayName'], row['description'], row['synonyms']]
    else:
        parts = [row['displayName'], row['description']]

    parts_clean = [str(p).strip() for p in parts if pd.notna(p)]
    return " ".join(parts_clean)

# Aplica esta función a los nodos
nodes_df['embedding_text'] = nodes_df.apply(generate_node_text, axis=1)

#Paso 3: Crear el objeto HeteroData con tokenización
#Ahora generaremos los tokens (en lugar de embeddings) y los guardaremos directamente en cada nodo:

data = HeteroData()

node_types = ["Drug", "Disorder", "Protein", "Gene"]
for ntype in node_types:
    data[ntype].input_ids = []
    data[ntype].attention_mask = []
    data[ntype].original_id = []

id_to_type = {}

# Tokenización
for _, row in nodes_df.iterrows():
    node_id = str(row['primaryDomainId'])
    node_type = row['type']

    if node_type not in node_types:
        continue

    text = row['embedding_text']
    encoded_input = tokenizer(text, padding='max_length', truncation=True,
                              max_length=128, return_tensors='pt')

    data[node_type].input_ids.append(encoded_input['input_ids'].squeeze(0))
    data[node_type].attention_mask.append(encoded_input['attention_mask'].squeeze(0))
    data[node_type].original_id.append(node_id)

    id_to_type[node_id] = node_type

# Convertir listas a tensores
for ntype in node_types:
    data[ntype].input_ids = torch.stack(data[ntype].input_ids)
    data[ntype].attention_mask = torch.stack(data[ntype].attention_mask)
    data[ntype].id = torch.arange(data[ntype].input_ids.size(0))
    print(f"[INFO] {ntype}: {data[ntype].input_ids.shape[0]} nodos tokenizados.")


#En el paso anterior se han guardado directamente los tokens en vez de embeddings pre-calculados.
#Paso 4: Añadir edges al grafo
#Esto es igual que antes:

edge_dict = {}

# Para mapear original_id (string) → posición (índice)
original_id_to_idx = {
    ntype: {id_: idx for idx, id_ in enumerate(data[ntype].original_id)}
    for ntype in node_types
}

for _, row in edges_df.iterrows():
    src_id = str(row['sourceDomainId'])
    tgt_id = str(row['targetDomainId'])
    relation = row['type']

    src_type = id_to_type.get(src_id)
    tgt_type = id_to_type.get(tgt_id)

    if src_type is None or tgt_type is None:
        continue

    key = (src_type, relation, tgt_type)
    if key not in edge_dict:
        edge_dict[key] = [[], []]

    # Usamos diccionario original_id_to_idx para obtener índices
    src_idx = original_id_to_idx[src_type].get(src_id)
    tgt_idx = original_id_to_idx[tgt_type].get(tgt_id)

    if src_idx is None or tgt_idx is None:
        continue

    edge_dict[key][0].append(src_idx)
    edge_dict[key][1].append(tgt_idx)

# Guardar edges en HeteroData
for (src_type, relation, tgt_type), (src_indices, tgt_indices) in edge_dict.items():
    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
    data[(src_type, relation, tgt_type)].edge_index = edge_index
    print(f"[INFO] {edge_index.size(1)} edges añadidos: ({src_type}, {relation}, {tgt_type}).")


from torch_geometric.transforms import ToUndirected
data = ToUndirected()(data)

#Paso 6: División en Train, Val y Test (Link prediction)

from torch_geometric.transforms import RandomLinkSplit

# Eliminar listas que PyG no puede manejar
for node_type in data.node_types:
    del data[node_type].original_id

# Añade explícitamente num_nodes
for node_type in data.node_types:
    data[node_type].num_nodes = data[node_type].input_ids.size(0)

transform = RandomLinkSplit(
    edge_types=[("Drug", "DrugHasIndication", "Disorder")],
    rev_edge_types=[("Disorder", "rev_DrugHasIndication", "Drug")],
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=2.0,
)

train_data, val_data, test_data = transform(data)

from torch_geometric.loader import LinkNeighborLoader

# Train loader
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[5, 2],  # vecinos por capa (correcto para empezar)
    edge_label_index=("Drug", "DrugHasIndication", "Disorder"),
    edge_label=train_data["Drug", "DrugHasIndication", "Disorder"].edge_label,
    batch_size=8,  # ajusta según tu GPU
    shuffle=True,
)

# Val loader
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[5, 2],
    edge_label_index=("Drug", "DrugHasIndication", "Disorder"),
    edge_label=val_data["Drug", "DrugHasIndication", "Disorder"].edge_label,
    batch_size=8,
    shuffle=False,
)

# Test loader (para evaluar al final)
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[5, 2],
    edge_label_index=("Drug", "DrugHasIndication", "Disorder"),
    edge_label=test_data["Drug", "DrugHasIndication", "Disorder"].edge_label,
    batch_size=8,
    shuffle=False,
)

def train_epoch_bce(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch)
        labels = batch["Drug", "DrugHasIndication", "Disorder"].edge_label.float().to(device)
        loss = F.binary_cross_entropy_with_logits(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_labels.append(labels.detach().cpu())
        all_preds.append(torch.sigmoid(preds).detach().cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())

    print(f"[INFO] Train Loss: {total_loss / len(loader):.4f}, Train AUC: {auc:.4f}")  # Imprime la pérdida y AUC

    return total_loss / len(loader), auc


@torch.no_grad()
def evaluate_bce(model, loader, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        labels = batch["Drug", "DrugHasIndication", "Disorder"].edge_label.float().to(device)
        loss = F.binary_cross_entropy_with_logits(preds, labels)

        total_loss += loss.item()
        all_labels.append(labels.cpu())
        all_preds.append(torch.sigmoid(preds).cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())

    return total_loss / len(loader), auc

def train_loop_bce(model, train_loader, val_loader, epochs=150, lr=5e-5, weight_decay=1e-5,
                   model_file="gnn_model_finetuned.pth", device='cuda', patience=15):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_val_auc = 0.0
    early_stop_counter = 0

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        print(f"[INFO] Epoch {epoch} iniciando...")
        train_loss, train_auc = train_epoch_bce(model, train_loader, optimizer, device)
        val_loss, val_auc = evaluate_bce(model, val_loader, device)
        scheduler.step(val_auc)

        print(f"Epoch {epoch:02d} | TrainLoss={train_loss:.4f}, TrainAUC={train_auc:.4f}, ValLoss={val_loss:.4f}, ValAUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_file)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("[INFO] Early stopping activado.")
                break

    model.load_state_dict(torch.load(model_file))
    return model


# Parámetros recomendados:
model = Model(dropout=0.3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("[INFO] Inicio del entrenamiento...")

trained_model = train_loop_bce(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=150,                   
    lr=5e-5,                      
    weight_decay=1e-5,            
    model_file="gnn_model_finetuned.pth",
    device=device,
    patience=15                   
)

print("[INFO] Entrenamiento terminado correctamente.")