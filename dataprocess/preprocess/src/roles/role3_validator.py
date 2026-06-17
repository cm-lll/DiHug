# preprocess/src/roles/role3_validator.py
import json
from preprocess.config.settings import settings

class Role3Validator:
    def __init__(self, llm_client, system_prompt: str, role_prompt: str):
        self.llm = llm_client
        self.system_prompt = system_prompt + "\n" + role_prompt
        self.model = settings.ROLE3_MODEL

    def build_prompt(self, sample_graph, canonical_map_sample):
        return f"Graph sample:\n{json.dumps(sample_graph, ensure_ascii=False, indent=2)}\n\nCanonical sample:\n{json.dumps(canonical_map_sample, ensure_ascii=False, indent=2)}\n\nReturn JSON: {{'warnings':[], 'fixes':[]}}"

    def validate(self, nodes, edges, canonical_map, sample_limit=500):
        sample_nodes = nodes[:sample_limit]
        sample_edges = edges[:sample_limit]
        sample_map = {k: canonical_map[k] for k in list(canonical_map)[:200]} if canonical_map else {}
        prompt = self.build_prompt({"nodes": sample_nodes, "edges": sample_edges}, sample_map)
        resp_text = self.llm.ask(prompt=prompt, model=self.model, system_prompt=self.system_prompt)
        try:
            out = json.loads(resp_text)
        except Exception:
            out = {"warnings": ["invalid json from LLM; raw: "+resp_text[:300]], "fixes": []}
        return out
