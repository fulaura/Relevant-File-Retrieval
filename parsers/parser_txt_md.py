def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return {"text": text.strip(), "tables": []}
