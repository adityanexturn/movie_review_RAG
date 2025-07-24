import os
import json
import pandas as pd
import nltk
from PyPDF2 import PdfReader

# Ensure required sentence tokenizer model is downloaded (run once)
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def clean_dataframe(df):
    """
    Cleans a pandas DataFrame by removing all-NaN columns and
    attempting to exclude purely numeric columns to focus on textual content.
    Returns a string where each row is joined, separated by newlines.
    """
    df = df.dropna(how='all', axis=1)
    # Select columns that are likely to be informative (not purely numeric)
    selected_cols = [
        col for col in df.columns
        if df[col].dtype == object or not df[col].astype(str).str.match(r'^\d+(\.\d+)?$').all()
    ]
    if not selected_cols:
        selected_cols = df.columns
    # Concatenate all selected columns for each row into a single line
    lines = df[selected_cols].astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
    return "\n".join(lines)

def load_text_file(filepath):
    """
    Loads the content of a .txt file as a single string.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def load_json_file(filepath):
    """
    Loads and flattens JSON data into a continuous string.
    Handles both dictionary and list top-level structures.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return " ".join(str(v) for v in data.values())
    elif isinstance(data, list):
        return " ".join(str(item) for item in data)
    return str(data)

def load_excel_file(filepath):
    """
    Loads an Excel file, cleans it, and returns as a newline-separated string.
    """
    df = pd.read_excel(filepath)
    return clean_dataframe(df)

def load_csv_file(filepath):
    """
    Loads a CSV file using various encodings for compatibility.
    Cleans it to remove numeric/noisy columns and returns as newline-separated string.
    """
    for encoding in ['utf-8', 'cp1252', 'latin1']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Unable to read {filepath} with tried encodings")
    return clean_dataframe(df)

def load_pdf_file(filepath):
    """
    Extracts text from all pages of a PDF file and concatenates as a single string.
    If a page contains no text, it is skipped.
    """
    reader = PdfReader(filepath)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def chunk_text(text):
    """
    Splits a large text into individual sentences using NLTK's sentence tokenizer.
    """
    return sent_tokenize(text)

def load_all_documents(folder_path):
    """
    Loads all supported files from the specified folder,
    processes each file according to its format, and splits
    the text into meaningful chunks for downstream NLP tasks.
    Structured files are split line-wise, while prose is split into sentences.
    Returns a list of dictionaries, each representing one chunk.
    """
    documents = []

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if not os.path.isfile(path):
            continue

        # File format handlers
        if filename.endswith('.txt'):
            text = load_text_file(path)
        elif filename.endswith('.json'):
            text = load_json_file(path)
        elif filename.endswith('.xlsx'):
            text = load_excel_file(path)
        elif filename.endswith('.csv'):
            text = load_csv_file(path)
        elif filename.endswith('.pdf'):
            text = load_pdf_file(path)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        # For structured data, split by line; for others, use sentence tokenization
        if filename.endswith(('.csv', '.xlsx')):
            lines = text.split('\n')
            chunks = [line.strip() for line in lines if line.strip()]
        else:
            chunks = chunk_text(text)

        # For each chunk, create a dictionary with associated metadata
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk.strip(),
                "id": f"{filename}_{i}",
                "source": filename
            })

    return documents

if __name__ == "__main__":
    # Update this path to match your project structure
    folder_path = r"C:\Users\adity\Desktop\Gen-Ai Rag\review-data"

    print("Loading documents from:", folder_path)
    docs = load_all_documents(folder_path)
    print(f"\nLoaded {len(docs)} chunks total.")
    print("First 3 chunks:\n")
    for i, doc in enumerate(docs[:3]):
        print(f"{i+1}. [{doc['source']}] → {doc['text'][:100]}...\n")
