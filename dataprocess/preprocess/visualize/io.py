import json

def read_jsonl_in_chunks(path, chunk_size):
    chunk = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip(): continue
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def write_jsonl(path, iterable):
    with open(path, "w", encoding="utf8") as f:
        for d in iterable:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
