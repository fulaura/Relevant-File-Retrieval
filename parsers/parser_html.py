from bs4 import BeautifulSoup
from pathlib import Path
import os
from datetime import datetime

def parse_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Извлекаем обычный текст
    text = soup.get_text(separator="\n", strip=True)

    # Извлекаем таблицы
    tables = []
    for table in soup.find_all("table"):
        parsed_table = []
        for row in table.find_all("tr"):
            parsed_row = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            parsed_table.append(parsed_row)
        tables.append(parsed_table)

    # Файловая информация
    file_name = Path(file_path).stem
    file_type = "html"
    stat = os.stat(file_path)
    created_date = datetime.fromtimestamp(stat.st_ctime)
    modified_date = datetime.fromtimestamp(stat.st_mtime)

    return {
        "file_name": file_name,
        "file_type": file_type,
        "created_date": created_date,
        "modified_date": modified_date,
        "content": text.strip(),
        "metadata": None,
        "path": str(Path(file_path).resolve()),
        "embedding": None
    }

if __name__ == "__main__":
    # Example usage
    file_path = 'test/sample.html'
    parsed_data = parse_html(file_path)
    print(parsed_data)
