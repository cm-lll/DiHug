# preprocess/src/llm_client.py
import time
import traceback
from google import genai
from google.genai import types
from preprocess.config.settings import settings


def read_file(path: str):
    """Load prompt file content instead of passing file path."""
    if not path:
        return ""
    with open(path, "r", encoding="utf8") as f:
        return f.read()


class LLMClient:
    """
    Gemini ChatSession wrapper.
    - system_instruction is passed ONCE per session
    - supports auto session rotation
    - prints prompt + response for debugging
    """

    SESSION_MAX_USES = 30  # avoid drift

    def __init__(self, api_key=None, system_prompt=None):
        key = api_key or settings.API_KEY
        if not key:
            raise ValueError("Missing GEMINI_API_KEY")

        self.client = genai.Client(api_key=key)
        self.system_instruction = [system_prompt] if system_prompt else []

        self.current_model = settings.ROLE1_MODEL
        self.chat = self._new_session(self.current_model)
        self.use_count = 0

    # -----------------------------------------
    def _new_session(self, model: str):
        """Create a fresh ChatSession with persisted system_instruction."""
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction
        )
        return self.client.chats.create(model=model, config=config)

    # -----------------------------------------
    def ask(self, prompt: str, model: str = None):
        """
        Send user prompt to Gemini with automatic:
        - model switching
        - session rebuilding
        - prompt/response debug logging
        """

        # Model changed → reset session
        if model and model != self.current_model:
            self.current_model = model
            self.chat = self._new_session(model)
            self.use_count = 0

        # Rotate session every N messages
        if self.use_count >= self.SESSION_MAX_USES:
            self.chat = self._new_session(self.current_model)
            self.use_count = 0

        try:
            resp = self.chat.send_message(prompt)
            self.use_count += 1
            return resp.text

        except Exception as e:
            print("\n[LLM ERROR] Rebuilding session...")
            print(traceback.format_exc())
            time.sleep(0.3)

            self.chat = self._new_session(self.current_model)
            self.use_count = 0

            # Retry once
            try:
                resp = self.chat.send_message(prompt)
                return resp.text
            except:
                return "[]"
