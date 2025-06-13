import os
import json
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components
from tkinter import filedialog, messagebox, Text
from threading import Thread
from tqdm import tqdm

class GHAC:
    def __init__(self):
        pass

    @staticmethod
    def load_gcn_embeddings(embedding_path):
        """Loads GCN embeddings from a PKL file."""
        try:
            with open(embedding_path, "rb") as file_obj:
                return pickle.load(file_obj)
        except FileNotFoundError:
            print(f"Error: Embedding file not found at {embedding_path}.")
            return None

    @staticmethod
    def GHAC(embeddings, n_clusters=-1):
        """Performs clustering using Agglomerative Clustering."""
        if not embeddings:
            return []

        distance = pairwise_distances(embeddings, metric="cosine")

        model = AgglomerativeClustering(
            metric="precomputed", linkage="average", n_clusters=n_clusters
        ) if n_clusters != -1 else AgglomerativeClustering(metric="precomputed", linkage="average")

        return model.fit_predict(distance)

    @staticmethod
    def pairwise_evaluate(correct_labels, pred_labels):
        """Calculates pairwise precision, recall, and F1 for clustering evaluation."""
        TP, TP_FP, TP_FN = 0.0, 0.0, 0.0

        for i in range(len(correct_labels)):
            for j in range(i + 1, len(correct_labels)):
                if correct_labels[i] == correct_labels[j]:
                    TP_FN += 1
                if pred_labels[i] == pred_labels[j]:
                    TP_FP += 1
                if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                    TP += 1

        if TP == 0:
            return 0, 0, 0

        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

        return pairwise_precision, pairwise_recall, pairwise_f1

    @staticmethod
    def calculate_ACP_AAP(correct_labels, cluster_labels):
        """Computes ACP (Average Cluster Purity) and AAP (Average Author Purity)."""
        correct_labels = np.array(correct_labels)
        unique_clusters = np.unique(cluster_labels)
        ACP, AAP = 0.0, 0.0

        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_author_labels = correct_labels[cluster_indices]
            max_count = np.max(np.bincount(cluster_author_labels))
            ACP += max_count / len(cluster_indices)

        unique_authors = np.unique(correct_labels)
        for author in unique_authors:
            author_indices = np.where(correct_labels == author)[0]
            author_cluster_labels = cluster_labels[author_indices]
            max_count = np.max(np.bincount(author_cluster_labels))
            AAP += max_count / len(author_indices)

        ACP /= len(unique_clusters)
        AAP /= len(unique_authors)

        return ACP, AAP

    @staticmethod
    def calculate_KMetric(ACP, AAP):
        """Calculates the K-Metric based on ACP and AAP."""
        return np.sqrt(ACP * AAP)

    @staticmethod
    def cluster_evaluate(embedding_path, json_dir, result_box):
        """Processes JSON files and evaluates clustering results."""

        embeddings = GHAC.load_gcn_embeddings(embedding_path)
        if embeddings is None:
            result_box.insert("end", f"Error: Embeddings not found at {embedding_path}.\n")
            return

        results = []
        file_names = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        all_precision, all_recall, all_f1 = [], [], []
        all_acp, all_aap, all_k = [], [], []

        for fname in tqdm(file_names, desc="Processing JSON files"):
            with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as file:
                data = json.load(file)

            unique_labels = list(set(entry['label'] for entry in data))
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            correct_labels = [label_mapping[entry['label']] for entry in data]
            papers = [entry['id'] for entry in data]

            if len(correct_labels) < 2 or not papers:
                continue

            embeddings_data = [embeddings[f"i{pid}"] for pid in papers if f"i{pid}" in embeddings]
            if len(embeddings_data) == 0:
                continue

            predicted_labels = GHAC.GHAC(embeddings_data, n_clusters=len(set(correct_labels)))

            pairwise_precision, pairwise_recall, pairwise_f1 = GHAC.pairwise_evaluate(correct_labels, predicted_labels)
            ACP, AAP = GHAC.calculate_ACP_AAP(correct_labels, predicted_labels)
            K = GHAC.calculate_KMetric(ACP, AAP)

            all_precision.append(pairwise_precision)
            all_recall.append(pairwise_recall)
            all_f1.append(pairwise_f1)
            all_acp.append(ACP)
            all_aap.append(AAP)
            all_k.append(K)

            results.append([fname, pairwise_precision, pairwise_recall, pairwise_f1, ACP, AAP, K])

            result_box.insert("end", f"Processed: {fname}\n")
            result_box.see("end")

        avg_results = ["AVERAGE", np.mean(all_precision), np.mean(all_recall), np.mean(all_f1), np.mean(all_acp), np.mean(all_aap), np.mean(all_k)]
        results.append(avg_results)

        results_df = pd.DataFrame(results, columns=["Author", "Pairwise Precision", "Pairwise Recall", "Pairwise F1", "ACP", "AAP", "K"])
        output_path = os.path.join(json_dir, 'clustering_results.csv')
        results_df.to_csv(output_path, index=False)

        result_box.insert("end", f"\n===== AVERAGE RESULTS =====\n")
        result_box.insert("end", f"Precision: {avg_results[1]:.4f}, Recall: {avg_results[2]:.4f}, F1: {avg_results[3]:.4f}\n")
        result_box.insert("end", f"ACP: {avg_results[4]:.4f}, AAP: {avg_results[5]:.4f}, K: {avg_results[6]:.4f}\n")
        result_box.insert("end", f"\nResults saved at {output_path}.\n")
        result_box.see("end")


class GHACApp(ttk.Window):
    """GUI application for running GHAC clustering"""

    def __init__(self):
        super().__init__(themename="superhero")
        self.title("GCN Embedding Validation")
        self.geometry("800x600")

        self.embedding_path_var = ttk.StringVar()
        self.json_dir_var = ttk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=BOTH, expand=True)

        ttk.Label(frame, text="Embedding Path:", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        ttk.Entry(frame, textvariable=self.embedding_path_var, width=80).pack(pady=5)
        ttk.Button(frame, text="Select Embedding", command=lambda: self.embedding_path_var.set(filedialog.askopenfilename(filetypes=[("PKL File", "*.pkl")]))).pack(pady=5)

        ttk.Label(frame, text="JSON Directory:", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        ttk.Entry(frame, textvariable=self.json_dir_var, width=80).pack(pady=5)
        ttk.Button(frame, text="Select Directory", command=lambda: self.json_dir_var.set(filedialog.askdirectory())).pack(pady=5)

        ttk.Button(frame, text="Start Validation", command=self.start_clustering).pack(pady=20)

        self.result_box = Text(frame, wrap="word", height=15, width=80)
        self.result_box.pack(fill=BOTH, expand=True)

    def start_clustering(self):
        Thread(target=GHAC.cluster_evaluate, args=(self.embedding_path_var.get(), self.json_dir_var.get(), self.result_box)).start()


if __name__ == "__main__":
    app = GHACApp()
    app.mainloop()
