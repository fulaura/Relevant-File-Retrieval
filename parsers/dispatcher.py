from pathlib import Path
from parsers.parser_pdf import parse_pdf
from parsers.parser_docx import parse_docx
from parsers.parser_txt_md import parse_txt
from parsers.parser_html import parse_html


#TODO: implement a image parser/handler for each file type
def parse_document(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".txt", ".md"]:
        return parse_txt(file_path)
    elif ext in [".html", ".htm"]:
        return parse_html(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


