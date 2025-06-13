import json
import os
import re
from tkinter import Tk, filedialog, messagebox, Button, Label

# Function to sanitize the author's name to be used as a filename
def clean_filename(name):
    if not name or not isinstance(name, str):
        return "unknown_author"
    # Replace invalid characters with underscore
    return re.sub(r'[\/:*?"<>|]', '_', name.replace(' ', '_'))

# Main function to split the JSON by author
def split_json_by_author():
    # Tkinter window setup
    root = Tk()
    root.withdraw()  # Hide the main window

    # Select the input JSON file
    json_file = filedialog.askopenfilename(title="Select the input JSON file", filetypes=[("JSON files", "*.json")])
    if not json_file:
        messagebox.showerror("Error", "No JSON file selected.")
        return

    # Select the output directory
    output_dir = filedialog.askdirectory(title="Select the output directory")
    if not output_dir:
        messagebox.showerror("Error", "No output directory selected.")
        return

    output_dir = os.path.join(output_dir, 'authors_json')
    os.makedirs(output_dir, exist_ok=True)

    # Load the main JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load JSON file: {e}")
        return

    # Split the data by author and save individual JSONs
    for entry in data:
        author = entry.get('author')

        # Check if 'author' field is valid
        if author is None or not isinstance(author, str) or author.strip() == '':
            author = 'unknown_author'
        else:
            author = author.strip()

        author_filename = clean_filename(author) + '.json'
        json_output = os.path.join(output_dir, author_filename)

        # Append to existing file or create a new one
        if os.path.isfile(json_output):
            with open(json_output, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)

            existing_data.append(entry)

            with open(json_output, 'w', encoding='utf-8') as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
        else:
            with open(json_output, 'w', encoding='utf-8') as file:
                json.dump([entry], file, indent=4, ensure_ascii=False)

    messagebox.showinfo("Done", "JSON files have been created for each author.")
    print("✅ JSON files created for each author.")

if __name__ == "__main__":
    split_json_by_author()
