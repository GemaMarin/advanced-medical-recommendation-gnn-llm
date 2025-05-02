import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Cargar el CSV de los nodos limpio
nodes_df = pd.read_csv('nodes.csv', low_memory=False)

# Función para generar texto específico según el tipo de nodo
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

    # Eliminar NaNs y unir textos
    parts_clean = [str(p).strip() for p in parts if pd.notna(p)]
    return " ".join(parts_clean)

# Aplicar la función a todo el DataFrame
nodes_df['embedding_text'] = nodes_df.apply(generate_node_text, axis=1)

# Verificar algunos ejemplos generados
print(nodes_df[['type', 'embedding_text']].head())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased').to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

embeddings = []
texts = nodes_df['embedding_text'].tolist()

for text in tqdm(texts, desc="Generando embeddings mejorados"):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    embeddings.append(embedding.squeeze())

nodes_df['embeddings'] = embeddings

# Guardar embeddings generados
import pickle
data_dict = {
    'Nodes Name': nodes_df['primaryDomainId'].tolist(),
    'Nodes Display Name': nodes_df['displayName'].tolist(),
    'Category': nodes_df['type'].tolist(),
    'New Embedding': embeddings
}

with open('embeddings_mejorados_biobert.pkl', 'wb') as f:
    pickle.dump(data_dict, f)