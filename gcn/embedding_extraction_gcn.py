
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pickle
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Text
from threading import Thread

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load NLP-based embeddings from a .pkl file
def load_embeddings(embedding_path):
    try:
        with open(embedding_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except FileNotFoundError:
        print(f"❌ Error: Embedding file {embedding_path} not found.")
        return None


# Prepare features and edge index for the GCN model
def prepare_features(G, embeddings, device):
    nodes = list(G.nodes)
    node_idx_map = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_idx_map[u], node_idx_map[v]) for u, v in G.edges]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    if embeddings:
        sample_embedding = next(iter(embeddings.values()))
        embedding_dim = sample_embedding.shape[0]
        features = [embeddings.get(node, np.zeros(embedding_dim)) for node in nodes]
    else:
        embedding_dim = 128
        features = np.random.normal(loc=0.0, scale=1.0, size=(len(nodes), embedding_dim))

    x = torch.tensor(features, dtype=torch.float).to(device)
    return Data(x=x, edge_index=edge_index), nodes, embedding_dim


# Define the GCN model architecture
class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, 512))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(512, 512))

        self.convs.append(GCNConv(512, 512))
        self.fc = torch.nn.Linear(512, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.fc(x)
        return x


# Train the GCN model
def train_gcn(data, input_dim, num_layers, epochs, progress_bar, progress_label, output_box):
    model = GCN(input_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(epochs):
        loss = train()
        progress = ((epoch + 1) / epochs) * 100
        progress_bar["value"] = progress
        progress_bar.update()
        progress_label.config(text=f"⚙ Training: Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
        output_box.insert("end", f"Epoch {epoch + 1}, Loss: {loss.item():.4f}\n")
        output_box.see("end")


    model.eval()
    with torch.no_grad():
        new_embeddings = model(data).cpu().numpy()

    return new_embeddings


# Save the refined node embeddings
def save_embeddings(embeddings, nodes, save_path):
    embeddings_dict = {node: embeddings[idx] for idx, node in enumerate(nodes)}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as file_obj:
        pickle.dump(embeddings_dict, file_obj)


# GUI class using ttkbootstrap
class GCNApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("GCN Embedding Training")
        self.geometry("800x700")

        self.network_file_var = ttk.StringVar()
        self.embedding_file_var = ttk.StringVar()
        self.num_layers_var = ttk.IntVar(value=3)
        self.epochs_var = ttk.IntVar(value=1000)

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=BOTH, expand=True)

        # Heterogeneous graph file selection
        ttk.Label(frame, text="Heterogeneous Graph File (.pkl):", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        network_file_entry = ttk.Entry(frame, textvariable=self.network_file_var, width=50, bootstyle="primary")
        network_file_entry.pack(pady=5)
        ttk.Button(frame, text="Browse", bootstyle="info",
                   command=lambda: self.network_file_var.set(filedialog.askopenfilename(filetypes=[("PKL Files", "*.pkl")], title="Select Graph File"))).pack(pady=5)

        # NLP Embedding file selection
        ttk.Label(frame, text="NLP Embeddings File (.pkl):", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        embedding_file_entry = ttk.Entry(frame, textvariable=self.embedding_file_var, width=50, bootstyle="primary")
        embedding_file_entry.pack(pady=5)
        ttk.Button(frame, text="Browse", bootstyle="info",
                   command=lambda: self.embedding_file_var.set(filedialog.askopenfilename(filetypes=[("PKL Files", "*.pkl")], title="Select Embeddings File"))).pack(pady=5)

        # Number of GCN layers
        ttk.Label(frame, text="Number of Layers:", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        num_layers_entry = ttk.Entry(frame, textvariable=self.num_layers_var, width=10, bootstyle="primary")
        num_layers_entry.pack(pady=5)

        # Number of training epochs
        ttk.Label(frame, text="Number of Epochs:", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        epochs_entry = ttk.Entry(frame, textvariable=self.epochs_var, width=10, bootstyle="primary")
        epochs_entry.pack(pady=5)

        # Start training button
        ttk.Button(frame, text="Start Training", bootstyle="success", command=self.start_training).pack(pady=20)

        # Progress bar
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=500, mode="determinate", bootstyle="success")
        self.progress_bar.pack(pady=10)

        # Progress label
        self.progress_label = ttk.Label(frame, text="")
        self.progress_label.pack(pady=5)

        # Output log
        result_frame = ttk.Labelframe(frame, text="Output", bootstyle="primary", padding=10)
        result_frame.pack(fill=BOTH, expand=True, pady=10)

        self.output_box = Text(result_frame, wrap="word", height=15, width=80)
        self.output_box.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(result_frame, command=self.output_box.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.output_box.config(yscrollcommand=scrollbar.set)

    def start_training(self):
        Thread(target=self.run_gcn_training_gui).start()

    def run_gcn_training_gui(self):
        network_file = self.network_file_var.get()
        embedding_file = self.embedding_file_var.get()
        num_layers = self.num_layers_var.get()
        epochs = self.epochs_var.get()

        with open(network_file, 'rb') as file:
            G = pickle.load(file)

        embeddings = load_embeddings(embedding_file)
        data, nodes, input_dim = prepare_features(G, embeddings, device)
        new_embeddings = train_gcn(data, input_dim, num_layers, epochs, self.progress_bar, self.progress_label, self.output_box)

        save_embeddings(new_embeddings, nodes, embedding_file.replace(".pkl", "_gcn.pkl"))


if __name__ == "__main__":
    app = GCNApp()
    app.mainloop()