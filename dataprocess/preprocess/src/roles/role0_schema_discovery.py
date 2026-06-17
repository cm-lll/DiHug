# preprocess/src/roles/role0_schema_discovery.py
import json
import re
from preprocess.config.settings import settings


class Role0SchemaDiscovery:
    def __init__(self, llm_client):
        self.llm = llm_client

    def _compact_samples(self, node_type: str, samples: list) -> list:
        """
        Reduce prompt size for Role0. Role0 only needs signals for schema discovery.
        Keep a minimal set of fields by type.
        """
        compact = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            if node_type == "Paper":
                compact.append({
                    "id": s.get("id"),
                    "title": s.get("title", ""),
                    "keywords": s.get("keywords", [])[:10] if isinstance(s.get("keywords"), list) else [],
                    "venue": s.get("venue", ""),
                    "doc_type": s.get("doc_type", ""),
                    "year": s.get("year"),
                    "n_citation": s.get("n_citation", 0),
                })
            elif node_type == "Author":
                # IMPORTANT: expose quantitative signals
                paper_ids = s.get("paper_ids", [])
                paper_count = len(paper_ids) if isinstance(paper_ids, list) else (s.get("paper_count") or 0)
                compact.append({
                    "id": s.get("id"),
                    "name": s.get("name", ""),
                    "org_name": s.get("org_name", ""),
                    "paper_count": paper_count,
                    "citation_sum": s.get("citation_sum", 0),
                })
            elif node_type == "Organization":
                compact.append({
                    "id": s.get("id"),
                    "name": s.get("name", ""),
                    "raw_name": s.get("raw_name", ""),
                })
            else:
                # fallback
                compact.append({k: s.get(k) for k in ("id", "type", "name", "title") if k in s})
        return compact

    def _extract_json_object(self, text: str):
        """
        Robustly extract the first JSON object from a messy LLM response.
        Returns dict or None.
        """
        if not text:
            return None

        t = text.strip()

        # Remove code fences if any
        # Handles ```json ... ``` or ``` ... ```
        t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)

        # Fast path: direct JSON
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Fallback: find first {...} block (balanced-ish)
        # This is not perfect JSON parser, but works well for typical LLM outputs.
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            return None
        block = m.group(0).strip()
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def discover_schema(self, node_type, samples):
        """
        Discover subtype schema for a node category.
        Always returns a dict:
          {"should_classify": bool, "schema": list}
        """
        # reduce token usage for Role0
        samples = samples[:200]  # hard cap just in case
        compact_samples = self._compact_samples(node_type, samples)

        sample_text = json.dumps(compact_samples, ensure_ascii=False, indent=2)

        # Add explicit guidance for Author quantitative tiers
        extra_guidance = ""
        if node_type == "Author":
            extra_guidance = (
                "\nSpecial guidance for Author:\n"
                "- If sample nodes contain quantitative signals like paper_count and citation_sum,\n"
                "  prefer a coarse, reusable impact/activity tier schema such as:\n"
                '  ["HighImpact", "MidImpact", "LowImpact", "Unknown"]\n'
                "- Do NOT include numeric thresholds in the schema.\n"
            )

        prompt = (
            f"Node Category: {node_type}\n\n"
            f"Sample Nodes:\n{sample_text}\n\n"
            "Role0 Task: Discovering a Meaningful Subtype Schema for a Node Category\n"
            "Goal:\n"
            "Given a sample of nodes from the same category, infer a reasonable subtype schema that is:\n"
            "- meaningful for downstream heterogeneous graph learning,\n"
            "- general enough to apply beyond the current sample,\n"
            "- grounded in either the sample or widely accepted domain knowledge,\n"
            "- optional — propose it only if it truly makes sense.\n\n"
            "Output format (strict):\n"
            "{\n"
            '  "should_classify": true | false,\n'
            '  "schema": ["SubtypeA", "SubtypeB", ...]\n'
            "}\n\n"
            "Rules:\n"
            "1) Prefer general, reusable categories; avoid overly specific labels.\n"
            "2) You may propose subtype values even if they do not appear in the sample, if domain-relevant.\n"
            "3) If no meaningful axis exists, output should_classify=false and schema=[].\n"
            "4) Output ONLY the JSON object. No extra text.\n"
            f"{extra_guidance}\n"
        )

        # Default fallback
        fallback = {"should_classify": False, "schema": []}

        try:
            resp = self.llm.ask(prompt, model=settings.ROLE0_MODEL)
        except Exception as e:
            print(f"[WARN] Role0 LLM call failed for type={node_type}: {e}")
            return fallback

        obj = self._extract_json_object(resp)
        if not isinstance(obj, dict):
            print(f"[WARN] Role0 invalid JSON for type={node_type}. Raw head: {str(resp)[:120]}")
            return fallback

        # Normalize fields
        should = bool(obj.get("should_classify", False))
        schema = obj.get("schema", [])
        if not isinstance(schema, list):
            schema = []
        schema = [str(x).strip() for x in schema if str(x).strip()]

        return {"should_classify": should and len(schema) > 0, "schema": schema}
