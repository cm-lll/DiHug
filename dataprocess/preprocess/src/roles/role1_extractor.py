# preprocess/src/roles/role1_extractor.py
import json
import re
import os
from preprocess.config.settings import settings


class Role1Extractor:
    """
    Node subtype extraction guided by Role0 schema.
    """
    _DEBUG_PRINT_COUNT = 1   # class-level counter
    _DEBUG_PRINT_LIMIT = int(os.getenv("ROLE1_DEBUG_PRINT_LIMIT", "3"))

    def __init__(self, llm_client):
        self.llm = llm_client

    def _compact_node_for_llm(self, node: dict) -> dict:
        """
        Minimize node JSON for LLM:
        - remove long / irrelevant fields
        - keep only subtype-relevant signals
        """
        n = dict(node)  # shallow copy

        node_type = n.get("type")

        # ---- Author ----
        if node_type == "Author":
            # paper_ids -> paper_count
            pids = n.pop("paper_ids", None)
            if isinstance(pids, list):
                n["paper_count"] = len(pids)
            else:
                n["paper_count"] = int(n.get("paper_count", 0) or 0)

            # citation_sum is already numeric & useful
            n["citation_sum"] = int(n.get("citation_sum", 0) or 0)

        # ---- Paper ----
        elif node_type == "Paper":
            # truncate abstract aggressively if still exists
            abs_text = n.get("abstract")
            if isinstance(abs_text, str) and len(abs_text) > 300:
                n["abstract"] = abs_text[:300] + "..."

            # limit keywords
            kws = n.get("keywords")
            if isinstance(kws, list) and len(kws) > 10:
                n["keywords"] = kws[:10]

        # ---- Organization ----
        elif node_type == "Organization":
            # nothing heavy here; usually already compact
            pass

        # Remove fields that Role1 never needs
        n.pop("references", None)
        n.pop("edges", None)
        n.pop("role1_error", None)

        return n

    # parse code-block json
    def _parse_json_labels(self, text):
        cleaned = text.strip()
        cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            else:
                return [str(parsed)]
        except:
            pass

        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        return lines[:5] if lines else ["unknown"]

    def build_batch_prompt(self, nodes, schema=None):
        nodes_compact = [self._compact_node_for_llm(n) for n in nodes]

        p = "Nodes JSON array:\n"
        p += json.dumps(nodes_compact, ensure_ascii=False, indent=2)

        if schema is not None:
            p += "\n\nAllowed Subtypes:\n"
            p += json.dumps(schema, ensure_ascii=False, indent=2)

        # Reinforce strict single-label JSON output format
        p += (
            "\n\nOutput format (STRICT):\n"
            "- Return ONLY valid JSON.\n"
            "- Output MUST be a JSON array of length = number of input nodes.\n"
            "- Each element MUST be a JSON array containing exactly ONE string subtype, "
            "chosen from Allowed Subtypes.\n"
            'Example: [["SubtypeA"], ["SubtypeB"], ...]\n'
        )
        return p

    def extract_batch(self, nodes, schema=None):
        prompt = self.build_batch_prompt(nodes, schema)
        resp = self.llm.ask(prompt, model=settings.ROLE1_MODEL)

        cleaned = resp.strip()
        cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise RuntimeError("Role1 batch output is not a JSON array")
        # 允许模型少/多，长度不对就直接报错让上层 fallback
        if len(parsed) != len(nodes):
            raise RuntimeError(f"Role1 batch output length mismatch: got {len(parsed)} expected {len(nodes)}")

        # 每项强制收敛为单一子类别 ["X"]
        out = []
        for item in parsed:
            label = None

            # 标准格式: ["X"]
            if isinstance(item, list) and len(item) >= 1:
                label = str(item[0])

            # 偶尔模型直接给 "X"
            elif isinstance(item, str):
                label = item

            # 更鲁棒：如果返回了对象，尝试常见字段
            elif isinstance(item, dict):
                cand = item.get("label") or item.get("subtype")
                if isinstance(cand, str):
                    label = cand
                elif isinstance(cand, list) and cand:
                    label = str(cand[0])

            if not label:
                label = "unknown"  # 或者根据 schema 选择 "Other"/"Unknown"

            out.append([label])

        return out
