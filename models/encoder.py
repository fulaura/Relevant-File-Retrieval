# from time import sleep
# from IPython.display import clear_output
import torch
from typing import List, Sequence, Union
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(device)
model.eval()

def _normalize_input(text: str) -> str:
    return f"passage: {text.strip()}"

def _pool_embeddings(outputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    summed = (outputs * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, p=2, dim=1)

def _encode_batch(texts: Sequence[str]) -> torch.Tensor:
    prefixed = [_normalize_input(text) for text in texts]
    inputs = tokenizer(
        prefixed,
        return_tensors='pt',
        truncation=True,
        padding=True
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs).last_hidden_state
        embeddings = _pool_embeddings(outputs, inputs['attention_mask'])
    return embeddings

def generate_embedding(text: str) -> List[float]:
    if not isinstance(text, str) or not text.strip():
        return []
    return encode_text_transformer(text)

def generate_embedding_list(corpus: List[str], return_tensor: bool = True) -> Union[torch.Tensor, List[List[float]]]:
    valid_corpus = [text for text in corpus if isinstance(text, str) and text.strip()]
    if len(valid_corpus) != len(corpus):
        raise ValueError("All corpus entries must be non-empty strings.")
    if not valid_corpus:
        if return_tensor:
            return torch.empty((0, model.config.hidden_size))
        return []
    embeddings = _encode_batch(valid_corpus)
    return embeddings if return_tensor else embeddings.cpu().tolist()
    
##############################
    
def encode_text_transformer(text: str) -> List[float]:
    embeddings = _encode_batch([text])
    return embeddings[0].cpu().tolist()

def chunk_text(text, max_tokens=400, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

def prepare_chunks_for_db(document, model_encode):
    chunks = chunk_text(document["content"])
    rows = []
    for i, chunk in enumerate(chunks):
        embedding = model_encode(chunk)
        existing_metadata = document.get("metadata") or {}
        metadata = {
            **existing_metadata,
            "chunk_id": i,
            "total_chunks": len(chunks)
        }
        rows.append({
            "file_name": document["file_name"],
            "file_type": document["file_type"],
            "created_date": document["created_date"],
            "modified_date": document["modified_date"],
            "content": chunk,
            "metadata": metadata,
            "path": document["path"],
            "embedding": embedding
        })
    return rows








def _self_test():
    sample_doc = {
        "file_name": "example.txt",
        "file_type": "txt",
        "created_date": "2024-01-01 00:00:00",
        "modified_date": "2024-01-02 00:00:00",
        "content": (
            "President Biden met with the Kazakh delegation in Astana to discuss economic cooperation.\n"
            "The meeting covered energy, trade, and security topics.\n"
            "Both sides agreed to continue negotiations later in the year."
        ),
        "metadata": {"author": "Demo", "language": "en"},
        "path": "test/example.txt",
    }

    print("Encoding chunks...")
    chunks = prepare_chunks_for_db(sample_doc, encode_text_transformer)
    print(f"Chunks generated: {len(chunks)}")
    if chunks:
        print(f"First chunk length: {len(chunks[0]['content'].split())} words")
        print(f"Embedding size: {len(chunks[0]['embedding'])}")

    try:
        from models import ner as ner_module
    except ImportError:
        print("NER module unavailable (run via `python -m models.encoder` to include package context).")
        return

    print("\nRunning NER on chunks...")
    annotated = ner_module.ner_collection(chunks)
    entities = annotated[0]["metadata"].get("entities", []) if annotated else []
    print(f"Sample entities: {entities[:3]}")


if __name__ == "__main__":
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    _self_test()
