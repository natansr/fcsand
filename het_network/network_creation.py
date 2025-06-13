import os
import pickle
import networkx as nx
from tkinter import filedialog, Tk

class HeterogeneousNetworkGenerator:
    def __init__(self):
        self.G = nx.Graph()  # Rede NetworkX

    def read_data(self, dirpath, selected_features):
        # Carregar e processar arquivos de relacionamentos com base nas features selecionadas
        print("Lendo dados dos arquivos...")
        try:
            if 'word' in selected_features:
                # Documentos e Palavras
                with open(os.path.join(dirpath, "paper_word.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, word = toks
                            self.G.add_node(paper_id, type='paper')
                            self.G.add_node(word, type='word')
                            self.G.add_edge(paper_id, word, relationship='contains')

            if 'author' in selected_features:
                # Documentos e Autores
                with open(os.path.join(dirpath, "paper_author.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, author_id = toks
                            self.G.add_node(paper_id, type='paper')
                            self.G.add_node(author_id, type='author')
                            self.G.add_edge(paper_id, author_id, relationship='written_by')

            if 'venue_name' in selected_features:
                # Documentos e Conferências
                with open(os.path.join(dirpath, "paper_venue.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, conf_id = toks
                            self.G.add_node(paper_id, type='paper')
                            self.G.add_node(conf_id, type='conference')
                            self.G.add_edge(paper_id, conf_id, relationship='presented_at')

            if 'title' in selected_features:
                # Documentos e Títulos
                with open(os.path.join(dirpath, "paper_title.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t", 1)
                        if len(toks) == 2:
                            paper_id, title = toks
                            title_node_id = f'title_{paper_id}'

                            self.G.add_node(title_node_id, type='title', content=title)
                            self.G.add_edge(paper_id, title_node_id, relationship='has_title')

            if 'abstract' in selected_features:
                # Documentos e Abstracts
                with open(os.path.join(dirpath, "paper_abstract.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t", 1)
                        if len(toks) == 2:
                            paper_id, abstract = toks
                            abstract_node_id = f'abstract_{paper_id}'

                            self.G.add_node(abstract_node_id, type='abstract', content=abstract)
                            self.G.add_edge(paper_id, abstract_node_id, relationship='has_abstract')

            if 'author_names' in selected_features:
                # Documentos e Nomes de Autores
                with open(os.path.join(dirpath, "paper_author_names.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t", 1)
                        if len(toks) == 2:
                            paper_id, author_names = toks
                            author_names_node_id = f'author_names_{paper_id}'

                            self.G.add_node(author_names_node_id, type='author_names', content=author_names)
                            self.G.add_edge(paper_id, author_names_node_id, relationship='has_author_names')

            # Documentos e Organizações (se aplicável)
            org_path = os.path.join(dirpath, "paper_organization.txt")
            if 'organization' in selected_features and os.path.exists(org_path):
                with open(org_path, encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, organization = toks
                            org_node_id = f'org_{organization}'

                            self.G.add_node(org_node_id, type='organization', name=organization)
                            self.G.add_edge(paper_id, org_node_id, relationship='affiliated_with')

            print(f"Nós adicionados: {self.G.number_of_nodes()}")
            print(f"Arestas adicionadas: {self.G.number_of_edges()}")

        except Exception as e:
            print(f"Erro ao ler dados: {e}")

    def save_network(self, filepath):
        # Salvar o grafo para uso futuro
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(self.G, file)
            print("Rede heterogênea salva com sucesso.")
        except Exception as e:
            print(f"Erro ao salvar a rede: {e}")

def main(selected_features):
    root = Tk()
    root.withdraw()  # Ocultar a janela principal do Tkinter

    dirpath = filedialog.askdirectory(title="Selecione o diretório contendo os arquivos de entrada")
    if not dirpath:
        print("Nenhum diretório selecionado.")
        return

    filepath = filedialog.asksaveasfilename(title="Selecione o local para salvar a rede heterogênea", defaultextension=".pkl",
                                            filetypes=[("Arquivos PKL", "*.pkl")])
    if not filepath:
        print("Nenhum caminho de saída selecionado.")
        return

    hng = HeterogeneousNetworkGenerator()
    hng.read_data(dirpath, selected_features)
    hng.save_network(filepath)

if __name__ == "__main__":
    # Exemplo de chamada com features selecionadas
    selected_features = ['word', 'author', 'venue_name', 'title', 'abstract', 'author_names', 'organization']
    main(selected_features)
