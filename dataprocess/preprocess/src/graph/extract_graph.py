from .normalize import canonicalize_org_name, org_id_from_name

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if n <= 0:
        return ""
    return s if len(s) <= n else s[:n] + "..."

def extract_nodes_and_edges(paper_record, max_abs_chars: int = 600, max_keywords: int = 10):
    paper_id = paper_record.get("id")
    if not paper_id:
        return [], []

    nodes, edges = [], []

    paper_node = {
        "type": "Paper",
        "id": paper_id,
        "title": _truncate(paper_record.get("title", "") or "", 200),
        "abstract": _truncate(paper_record.get("abstract", "") or "", max_abs_chars),
        "keywords": (paper_record.get("keywords", []) or [])[:max_keywords],
        "year": paper_record.get("year"),
        "n_citation": int(paper_record.get("n_citation", 0) or 0),
        "venue": paper_record.get("venue") or "",
        "doc_type": paper_record.get("doc_type") or ""
    }
    nodes.append(paper_node)

    authors = paper_record.get("authors", []) or []
    for idx, a in enumerate(authors):
        aid = a.get("id")  # 没有ID就忽略（按你的口径）
        if not aid:
            continue

        name = (a.get("name") or "").strip()
        org_raw = (a.get("org") or "").strip()
        org_name = canonicalize_org_name(org_raw) if org_raw else ""

        author_node = {
            "type": "Author",
            "id": aid,
            "name": name,
            "org_name": org_name,
            "paper_ids": [paper_id],
            "citation_sum": int(paper_record.get("n_citation", 0) or 0)
        }
        nodes.append(author_node)

        if a.get("is_corresponding", False) or "correspond" in (a.get("role", "") or "").lower():
            subtype = "corresponding_author"
        elif idx == 0:
            subtype = "first_author"
        elif idx == 1:
            subtype = "second_author"
        else:
            subtype = "co_author"
        edges.append({"src": aid, "dst": paper_id, "type": "author_of", "subtype": subtype})

        if org_name:
            org_id = org_id_from_name(org_name)
            org_node = {
                "type": "Organization",
                "id": org_id,
                "name": org_name,
                "raw_name": org_raw
            }
            nodes.append(org_node)
            edges.append({"src": aid, "dst": org_id, "type": "affiliated_with"})

    for ref in paper_record.get("references", []) or []:
        edges.append({"src": paper_id, "dst": ref, "type": "cites"})

    return nodes, edges
