# preprocess/src/pipeline.py
from preprocess.src.roles.role0_schema_discovery import Role0SchemaDiscovery
from preprocess.src.roles.role1_extractor import Role1Extractor
from preprocess.src.roles.role2_normalizer import Role2Normalizer
from preprocess.src.llm_client import LLMClient
from preprocess.config.settings import settings


class Pipeline:
    """
    Pipeline with OPTIONAL initialization of Role0/1/2 LLM clients.
    This avoids unnecessary LLM session creation.
    """

    def __init__(self,
                 api_key=None,
                 use_role0=False,
                 use_role1=False,
                 use_role2=False):
        key = api_key or settings.API_KEY

        self.role0 = None
        self.role1 = None
        self.role2 = None

        if use_role0:
            llm0 = LLMClient(key, open(settings.ROLE0_PROMPT_PATH).read())
            self.role0 = Role0SchemaDiscovery(llm0)

        if use_role1:
            llm1 = LLMClient(key, open(settings.ROLE1_PROMPT_PATH).read())
            self.role1 = Role1Extractor(llm1)

        if use_role2:
            llm2 = LLMClient(key, open(settings.ROLE2_PROMPT_PATH).read())
            self.role2 = Role2Normalizer(llm2)

    # For stage2
    def run_stage2_normalize(self, stage1_nodes):
        if not self.role2:
            raise RuntimeError("Role2 not initialized.")
        return self.role2.normalize(stage1_nodes)
