import json
import re
import os
from tkinter import filedialog, Tk, messagebox
from tqdm import tqdm

def preprocess_data(input_dir, output_dir, selected_features):
    if not input_dir or not output_dir:
        raise ValueError("Caminhos de entrada e saída não podem ser vazios.")
    os.makedirs(output_dir, exist_ok=True)

    # Dicionários para mapeamentos
    authors_map = {}
    venues_map = {}
    word_map = {}
    keyid = 0

    # Regex para remover caracteres especiais
    r = r'[!“”"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~—～]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the']

    # Carrega todos os arquivos JSON do input_dir
    all_data = []
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".json"):
            json_path = os.path.join(input_dir, file_name)
            with open(json_path, 'r', encoding='utf-8') as json_file:
                try:
                    file_data = json.load(json_file)
                    all_data.extend(file_data)
                except json.JSONDecodeError:
                    print(f"Arquivo {file_name} não pôde ser lido como JSON e será ignorado.")

    if not all_data:
        raise ValueError("Nenhum dado válido encontrado nos arquivos JSON do diretório.")

    # Abre os arquivos de saída
    with open(os.path.join(output_dir, 'paper_author.txt'), 'w', encoding='utf-8') as f_author_id, \
         open(os.path.join(output_dir, 'paper_venue.txt'), 'w', encoding='utf-8') as f_venue_id, \
         open(os.path.join(output_dir, 'paper_word.txt'), 'w', encoding='utf-8') as f_word, \
         open(os.path.join(output_dir, 'paper_title.txt'), 'w', encoding='utf-8') as f_title, \
         open(os.path.join(output_dir, 'paper_author_names.txt'), 'w', encoding='utf-8') as f_author_names, \
         open(os.path.join(output_dir, 'paper_abstract.txt'), 'w', encoding='utf-8') as f_abstract, \
         open(os.path.join(output_dir, 'paper_venue_name.txt'), 'w', encoding='utf-8') as f_venue_name:

        # 1) Primeira passada: processa (exceto contagem final de palavras)
        for entry in tqdm(all_data, desc="Processando JSON(s)"):
            pid = entry.get('id')
            label = entry.get('label')  # se precisar, verifique se != None

            # Ignora se não tiver id
            if pid is None:
                continue

            title = str(entry.get('title', '')).strip()
            # "authors" é uma string com múltiplos nomes separados por vírgula
            authors_str = entry.get('authors', '')  # pode vir vazio
            # Caso exista, separamos por vírgula
            all_authors = [a.strip() for a in authors_str.split(',') if a.strip()]

            # Observação: se não existir 'abstract' no JSON, ele virá vazio
            abstract = str(entry.get('abstract', '')).strip()
            # Seu JSON usa "venue" em vez de "conf"
            venue = str(entry.get('venue', '')).strip()

            # Se 'abstract' estiver na lista de features, escrevemos
            if 'abstract' in selected_features and abstract:
                f_abstract.write(f'i{pid}\t{abstract}\n')

            # Processa autores
            if all_authors:
                for author_name in all_authors:
                    # Normaliza o nome removendo espaços para indexar
                    author_clean = author_name.replace(" ", "")
                    if author_clean not in authors_map:
                        authors_map[author_clean] = keyid
                        keyid += 1
                    # Escreve: paper -> author_id
                    f_author_id.write(f'i{pid}\t{authors_map[author_clean]}\n')

                # Escreve os nomes (inteiros) no paper_author_names.txt
                f_author_names.write(f'i{pid}\t{",".join(all_authors)}\n')

            # Se 'venue_name' estiver selecionado, processamos o venue
            if 'venue_name' in selected_features and venue:
                if venue not in venues_map:
                    venues_map[venue] = keyid
                    keyid += 1
                # Escreve no paper_venue.txt (id) e paper_venue_name.txt (nome direto)
                f_venue_id.write(f'i{pid}\t{venues_map[venue]}\n')
                f_venue_name.write(f'i{pid}\t{venue}\n')

            # Limpa o título e escreve
            title_cleaned = re.sub(r, ' ', title).replace('\t', ' ').lower()
            f_title.write(f'i{pid}\t{title_cleaned}\n')

            # Se 'word' estiver selecionado, vamos fazer a contagem
            if 'word' in selected_features and title_cleaned:
                split_title = title_cleaned.split()
                for w in split_title:
                    if w and w not in stopword:
                        word_map[w] = word_map.get(w, 0) + 1

        # 2) Segunda passada: grava apenas as palavras que aparecem >= 2 vezes
        if 'word' in selected_features:
            for entry in tqdm(all_data, desc="Escrevendo palavras"):
                pid = entry.get('id')
                if pid is None:
                    continue

                title = str(entry.get('title', '')).strip()
                title_cleaned = re.sub(r, ' ', title).replace('\t', ' ').lower()
                split_title = title_cleaned.split()

                for w in split_title:
                    if w in word_map and word_map[w] >= 2:
                        f_word.write(f'i{pid}\t{w}\n')

    # Mensagem final
    messagebox.showinfo("Concluído", "Processamento de múltiplos JSON concluído com sucesso!")


def run_pre_processing(input_dir, output_dir, selected_features):
    if selected_features is None:
        selected_features = []
    preprocess_data(input_dir, output_dir, selected_features)
