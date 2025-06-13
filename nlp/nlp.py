
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import os
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Text, BooleanVar, StringVar
from ttkbootstrap.widgets import Checkbutton
from threading import Thread


class EmbeddingExtractor:
    def __init__(self, model_path, data_dir, selected_features, progress_bar, status_label, output_box, progress_label):
        self.model_path = model_path
        self.data_dir = data_dir
        self.selected_features = selected_features
        self.progress_bar = progress_bar
        self.status_label = status_label
        self.output_box = output_box
        self.progress_label = progress_label

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print_output(f"🖥️ Device selected: {self.device}")

        self._load_model()

    def _load_model(self):
        # Load a pretrained or fine-tuned model from directory or web.
        if os.path.isdir(self.model_path) and os.path.isfile(os.path.join(self.model_path, "config.json")):
            self.print_output(f"📥 Loading fine-tuned model from: {self.model_path}")
        else:
            self.print_output(f"📥 Downloading default model: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    def clean_value(self, value, substitute='.'):
        # Cleans null or empty values from input files.
        return substitute if value is None or str(value).lower() == 'null' else str(value).strip()

    def load_features(self):
        # Load features from input files.
        feature_files = {
            "abstract": "paper_abstract.txt",
            "author_names": "paper_author_names.txt",
            "title": "paper_title.txt",
            "venue_name": "paper_venue_name.txt",
            "word": "paper_word.txt",
        }

        data_dict = {feature: {} for feature in self.selected_features}

        for feature in self.selected_features:
            filename = feature_files.get(feature)
            file_path = os.path.join(self.data_dir, filename)

            if filename and os.path.exists(file_path):
                with open(file_path, encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            data_dict[feature][toks[0]] = self.clean_value(toks[1])

        return data_dict

    def combine_features(self, data_dict):
        # Combine selected features for embedding extraction.
        documents = []
        paper_ids = []

        title_dict = data_dict.get("title", {})

        for paperid in title_dict:
            combined_text = " ".join([data_dict[feature].get(paperid, "") for feature in self.selected_features]).strip()

            if combined_text and not combined_text.isspace():
                documents.append(combined_text)
                paper_ids.append(paperid)
            else:
                self.print_output(f"⚠ Warning: Empty document for paper ID {paperid}")

        return documents, paper_ids

    def get_embedding(self, text):
        # Extract embedding from text using loaded NLP model.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def extract_embeddings(self):
        # Run the embedding extraction process and save results.
        self.print_output(f"📥 Using model: {self.model_path}")
        self.print_output(f"🔹 Selected features: {', '.join(self.selected_features)}")

        data_dict = self.load_features()
        documents, paper_ids = self.combine_features(data_dict)

        if not documents:
            self.print_output("⚠ No documents available for embedding extraction. Please check input files.")
            return

        paper_vec = {}
        total_docs = len(documents)
        self.print_output(f"📄 Total documents to process: {total_docs}")
        self.progress_bar["maximum"] = total_docs

        for i, doc in enumerate(documents):
            paper_vec[paper_ids[i]] = self.get_embedding(doc)

            self.progress_bar["value"] = i + 1
            self.progress_bar.update_idletasks()
            self.progress_label.config(text=f"📄 Processing document {i + 1} of {total_docs}")

        save_path = os.path.join(self.data_dir, f"{os.path.basename(self.model_path).replace('/', '_')}_emb.pkl")
        os.makedirs(self.data_dir, exist_ok=True)

        with open(save_path, "wb") as file_obj:
            pickle.dump(paper_vec, file_obj)

        self.print_output(f"✅ Extraction complete and saved to: {save_path}")
        self.status_label.config(text="✅ Extraction completed!")

    def print_output(self, message):
        # Display messages in the output box and console.
        if self.output_box:
            self.output_box.insert("end", message + "\n")
            self.output_box.see("end")
        print(message)


def main():
    root = ttk.Window(themename="superhero")
    root.title("Embedding Extraction Progress")
    root.geometry("600x1000")

    ttk.Label(root, text="Select NLP model or fine-tuned directory:", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
    model_var = StringVar(value="allenai/scibert_scivocab_uncased")

    ttk.Combobox(root, textvariable=model_var, values=[
        "bert-base-uncased",
        "bert-large-uncased",
        "allenai/scibert_scivocab_uncased",
        "allenai/specter",
        "sentence-transformers/all-MiniLM-L6-v2"
    ], width=50, bootstyle="primary").pack(pady=5)


    ttk.Button(root, text="Select Fine-Tuned Model Directory", bootstyle="info",
               command=lambda: model_var.set(filedialog.askdirectory(title="Select fine-tuned model directory"))).pack(pady=5)

    selected_features = {feature: BooleanVar(value=True) for feature in ["abstract", "author_names", "title", "venue_name", "word"]}

    ttk.Label(root, text="Choose features for extraction:", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
    for feature, var in selected_features.items():
        Checkbutton(root, text=feature, variable=var, bootstyle="primary").pack(anchor="w")

    data_dir_var = StringVar()
    ttk.Button(root, text="Select Data Directory", bootstyle="info",
               command=lambda: data_dir_var.set(filedialog.askdirectory(title="Select data directory"))).pack(pady=5)

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate", bootstyle="success")
    progress_bar.pack(pady=10)

    ttk.Button(root, text="Start Extraction", bootstyle="success", command=lambda: Thread(
        target=EmbeddingExtractor(model_var.get(), data_dir_var.get(),
                                  [f for f, v in selected_features.items() if v.get()],
                                  progress_bar, ttk.Label(root), Text(root), ttk.Label(root)).extract_embeddings).start()).pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    main()
