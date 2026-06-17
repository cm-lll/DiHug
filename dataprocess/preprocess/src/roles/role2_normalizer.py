# preprocess/src/roles/role2_normalizer.py
import json
import re
from collections import defaultdict
from preprocess.config.settings import settings


class Role2Normalizer:

    def __init__(self, llm_client):
        self.llm = llm_client

    def _parse_json(self, text):
        cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except:
            return {}

    def normalize(self, stage1_nodes):
        # collect all strings
        vocab = set()
        for n in stage1_nodes:
            for s in n.get("raw_labels", []):
                vocab.add(s)

        prompt = "Raw Vocabulary:\n" + json.dumps(list(vocab), ensure_ascii=False, indent=2)
        prompt += "\n\nReturn a JSON mapping: raw_label -> canonical_label."

        resp = self.llm.ask(prompt, model=settings.ROLE2_MODEL)
        mapping = self._parse_json(resp)

        return mapping
