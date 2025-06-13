
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import Tk, filedialog, simpledialog, messagebox, BooleanVar, Checkbutton, Menu, Text

import het_network.network_creation as network_creation
import gcn.embedding_extraction_gcn as embedding_extraction_gcn

import subprocess
from threading import Thread

from data_process.pre_process_ghac import split_json_by_author
from data_process.pre_processing import run_pre_processing 


class MainApplication(ttk.Window):
    def __init__(self):
        super().__init__(themename="vapor")
        self.title("FCSAND ")
        self.geometry("600x500")
        
        self.selected_features = {
            "abstract": BooleanVar(value=True),
            "author_names": BooleanVar(value=True),
            "title": BooleanVar(value=True),
            "venue_name": BooleanVar(value=True),
            "word": BooleanVar(value=True)
        }

        self.setup_ui()
        self.create_menu()
    # Setup the main user interface with buttons and labels
    def setup_ui(self):
        ttk.Label(self, text="Welcome!", font=("Helvetica", 16)).pack(pady=10)

        ttk.Label(self, text="Select features for AND task:").pack(pady=5)
        for feature, var in self.selected_features.items():
            cb = ttk.Checkbutton(self, text=feature, variable=var, bootstyle="primary")
            cb.pack(anchor='w')

        ttk.Button(self, width=50, text="Input and Preprocessing", command=self.run_pre_processing, bootstyle="info").pack(pady=5)
        ttk.Button(self, width=50, text="Network Graph Creation", command=self.run_network_creation, bootstyle="success").pack(pady=5)
        ttk.Button(self, width=50, text="NLP Embeddings Extraction", command=self.run_embedding_extraction, bootstyle="success").pack(pady=5)
        ttk.Button(self, width=50, text="GCN", command=self.run_gcn_extraction, bootstyle="warning").pack(pady=5)
        ttk.Button(self, width=50, text="Clustering Preparation", command=self.run_dividir_json_por_autor, bootstyle="secondary").pack(pady=5)
        ttk.Button(self, width=50, text="Clustering with GHAC", command=self.run_clustering_validation, bootstyle="secondary").pack(pady=5)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate", bootstyle="warning")
        self.progress.pack(pady=20)

    # Create the top menu bar with options for file, theme, and help
    def create_menu(self):
        menu_bar = Menu(self)

        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        view_menu = Menu(menu_bar, tearoff=0)
        themes = ["solar", "darkly", "litera"]
        for theme in themes:
            view_menu.add_command(label=f"Theme {theme.capitalize()}", command=lambda t=theme: self.change_theme(t))
        menu_bar.add_cascade(label="View", menu=view_menu)

        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Expected JSON Format", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menu_bar)

    def show_about(self):
        about_message = "ComMAND v.1.4\n"
        messagebox.showinfo("About", about_message)

    def show_help(self):
        help_message = (
            "Expected JSON format:\n\n"
            "[\n"
            "  {\n"
            "    \"id\": <integer>,\n"
            "    \"label\": <integer>,\n"
            "    \"author\": \"Author Name\",\n"
            "    \"title\": \"Paper Title\",\n"
            "    \"venue\": \"Publication Venue\",\n"
            "    \"abstract\": \"Abstract Text\",\n"
            "    \"coauthors\": [\"coauthor1\", \"coauthor2\", ...]\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Ensure all required fields are filled and follow the correct format."
        )
        messagebox.showinfo("Expected JSON Format", help_message)

    def open_file(self):
        filedialog.askopenfile

    def change_theme(self, theme_name):
        self.style.theme_use(theme_name)

    def get_selected_features(self):
        selected = [feature for feature, var in self.selected_features.items() if var.get()]
        if not selected:
            messagebox.showwarning("Warning", "No feature selected. Please select at least one.")
        return selected

    # Run pre-processing on a selected input JSON directory
    def run_pre_processing(self):
        self.progress.start(10)
        try:
            input_dir = filedialog.askdirectory(title="Select directory with JSON files")
            if not input_dir:
                messagebox.showwarning("Warning", "No input directory selected.")
                return

            output_dir = filedialog.askdirectory(title="Select output directory")
            if not output_dir:
                messagebox.showwarning("Warning", "No output directory selected.")
                return

            selected_features = self.get_selected_features()
            if not selected_features:
                return

            run_pre_processing(input_dir, output_dir, selected_features)

        except Exception as e:
            messagebox.showerror("Error", f"Pre-processing failed: {e}")
        finally:
            self.progress.stop()
    # Create a heterogeneous graph using the selected features
    def run_network_creation(self):
        self.progress.start(10)
        try:
            selected_features = self.get_selected_features()
            if not selected_features:
                messagebox.showwarning("Warning", "No feature selected. Please select at least one.")
                return

            network_creation.main(selected_features)
            messagebox.showinfo("Success", "Heterogeneous graph created successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create graph: {e}")
        finally:
            self.progress.stop()
    # Launch external script to extract textual embeddings
    def run_embedding_extraction(self):
        subprocess.run(["python", "nlp/nlp.py"])
    # Launch GCN model training script
    def run_gcn_extraction(self):
        try:
            subprocess.run(["python", "gcn/embedding_extraction_gcn.py"])
        except Exception as e:
            messagebox.showerror("Error", f"GCN training failed: {e}")

    # Split JSON dataset into individual author files for GHAC validation (sometimes no needed)
    def run_dividir_json_por_autor(self):
        try:
            Thread(target=split_json_by_author).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to split JSON: {e}")
    # Launch GHAC clustering validation script
    def run_clustering_validation(self):
        try:
            subprocess.run(["python", "ghac/ghac.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Clustering execution failed: {e}")


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
