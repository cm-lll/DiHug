import os

class Settings:
    # Gemini API key for optional LLM-assisted preprocessing.
    # Set GEMINI_API_KEY in the environment; never commit secrets to the repo.
    API_KEY = os.getenv("GEMINI_API_KEY")

    # Input / output
    RAW_INPUT = os.getenv("RAW_INPUT", "preprocess/data/aminer_raw/aminer_subset.jsonl")
    OUT_STAGE1 = os.getenv("OUT_STAGE1", "preprocess/output/stage1_nodes.jsonl")
    OUT_EDGES  = os.getenv("OUT_EDGES", "preprocess/output/edges.jsonl")
    OUT_STAGE2 = os.getenv("OUT_STAGE2", "preprocess/output/stage2_refined.json")

    # multiprocess
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 20))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 5))

    # models
    ROLE0_MODEL = os.getenv("ROLE0_MODEL", "gemini-2.5-pro")
    ROLE1_MODEL = os.getenv("ROLE1_MODEL", "gemini-2.5-flash")
    ROLE2_MODEL = os.getenv("ROLE2_MODEL", "gemini-2.5-flash")

    # prompts
    ROLE0_PROMPT_PATH = os.getenv("ROLE0_PROMPT_PATH", "preprocess/prompts/role0_layer.txt")
    ROLE1_PROMPT_PATH  = os.getenv("ROLE1_PROMPT_PATH",  "preprocess/prompts/role1_layer.txt")
    ROLE2_PROMPT_PATH  = os.getenv("ROLE2_PROMPT_PATH",  "preprocess/prompts/role2_layer.txt")

settings = Settings()
