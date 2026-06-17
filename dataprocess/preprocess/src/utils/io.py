import json

def read_jsonl_in_chunks(path, chunk_size):
    chunk = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def write_jsonl(path, iterable_of_dicts, append=False):
    """
    Write iterable of dicts to a JSONL file.
    If append=True, append to file instead of overwriting.
    """
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf8") as f:
        for d in iterable_of_dicts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def load_text(path):
    """Load the content of a text file into a string."""
    with open(path, "r", encoding="utf8") as f:
        return f.read()