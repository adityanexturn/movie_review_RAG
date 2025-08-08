import os
import json
import pandas as pd
import nltk
from PyPDF2 import PdfReader

# --- CONFIGURATION ---
# Use a relative path. This works as long as the 'review-data' folder
# is in the same directory as your script. This is a best practice.
DATA_FOLDER = r"C:\Users\adity\Desktop\Gen-Ai Rag\review-data"

# Ensure required sentence tokenizer model is downloaded (run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading 'punkt' model for NLTK...")
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def clean_dataframe(df):
    """
    Cleans a pandas DataFrame by removing only completely empty columns.
    Preserves ALL data including numeric columns for comprehensive text extraction.
    Returns a string where each row is joined, separated by newlines.
    """
    # Only remove columns that are completely NaN
    df = df.dropna(how='all', axis=1)
    
    # Keep ALL columns - don't filter out numeric data
    # Convert everything to string and handle NaN values properly
    cleaned_df = df.fillna('').astype(str)
    
    # For each row, combine all column values with column names for context
    lines = []
    for _, row in cleaned_df.iterrows():
        # Create meaningful text by including column names with their values
        row_parts = []
        for col, value in row.items():
            if value and value.lower() != 'nan' and value.strip():  # Skip empty values
                row_parts.append(f"{col}: {value}")
        
        if row_parts:  # Only add non-empty rows
            lines.append(" | ".join(row_parts))
    
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
    Loads an Excel file, preserves ALL data including numeric columns.
    """
    df = pd.read_excel(filepath)
    return clean_dataframe(df)

def load_csv_file(filepath):
    """
    Loads a CSV file using various encodings for compatibility.
    Preserves ALL data including numeric columns.
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
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'. Please check the path.")
        return documents

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if not os.path.isfile(path):
            continue

        print(f"Processing: {filename}")

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
            if chunk.strip():  # Only add non-empty chunks
                documents.append({
                    "text": chunk.strip(),
                    "id": f"{filename}_{i}",
                    "source": filename
                })

    return documents

if __name__ == "__main__":
    print(f"Loading documents from folder: '{DATA_FOLDER}'")
    docs = load_all_documents(DATA_FOLDER)
    
    if docs:
        print(f"\nSuccessfully loaded {len(docs)} chunks total.")
        print("\n--- Sample of First 3 Chunks ---")
        for i, doc in enumerate(docs[:3]):
            print(f"{i+1}. [{doc['source']}] → {doc['text'][:150]}...")
    else:
        print("No documents were loaded. Please check your data folder and file formats.")
