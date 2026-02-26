import pdfplumber
import os
from pathlib import Path
from datetime import datetime

def parse_pdf(file_path):
    full_text = ""
    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() or ""
            table = page.extract_table()
            if table:
                tables.append(table)

    # Получаем имя файла
    file_name = Path(file_path).stem
    file_type = "pdf"
    
    # Получаем дату создания/модификации (система)
    stat = os.stat(file_path)
    created_date = datetime.fromtimestamp(stat.st_ctime)
    modified_date = datetime.fromtimestamp(stat.st_mtime)

    return {
        "file_name": file_name,
        "file_type": file_type,
        "created_date": created_date,
        "modified_date": modified_date,
        "content": full_text.strip(),
        "metadata": None,
        "path": str(Path(file_path).resolve()),
        "embedding": None
    }

if __name__ == "__main__":
    # Example usage
    file_path = 'test/A.Bekzat, K.Assem Conference paper 1.pdf'  # Replace with your PDF file path
    parsed_data = parse_pdf(file_path)
    print(parsed_data)