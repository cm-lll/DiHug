import json
from preprocess.src.pipeline import Pipeline
from preprocess.config.settings import settings
from preprocess.src.utils.io import read_jsonl_in_chunks

SAMPLE_PER_TYPE = 100

if __name__ == "__main__":
    # load raw
    sample_map = {"paper": [], "author": [], "org": []}

    count = {"paper":0, "author":0, "org":0}

    for chunk in read_jsonl_in_chunks(settings.RAW_INPUT, 2000):
        for item in chunk:
            for ntype in ["paper", "author", "org"]:
                if item.get("type") == ntype and len(sample_map[ntype]) < SAMPLE_PER_TYPE:
                    sample_map[ntype].append(item)
            # stop early if all full
        if all(len(sample_map[t]) >= SAMPLE_PER_TYPE for t in sample_map):
            break

    pipeline = Pipeline(api_key=settings.API_KEY, load_system_prompts=True, load_role_prompts=True)

    results = {}
    for ntype, samples in sample_map.items():
        results[ntype] = pipeline.role0.analyze(ntype, samples)

    with open("preprocess/output/schema_suggestion.json", "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Schema suggestion saved to preprocess/output/schema_suggestion.json")
