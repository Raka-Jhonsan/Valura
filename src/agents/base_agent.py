"""
BaseAgent — abstract base class all specialist agents inherit from.

Pattern mirrors the previous project's agent structure:
  - Class-level NAME and DESCRIPTION constants
  - self.log() instance method with colored prefix
  - run() is the single public method every agent must implement
  - run() is an async generator — yields string chunks for SSE streaming
"""
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from src.models import ClassifierOutput, UserProfile


class BaseAgent(ABC):
    """
    Every specialist agent subclasses this.
    Implement run() as an async generator that yields text chunks.
    """

    NAME: str = "base"
    DESCRIPTION: str = "Abstract base agent"

    # Subclasses override these for their colored log prefix
    LOG_COLOR: str = "\033[44m"  # BG_BLUE default
    LOG_RESET: str = "\033[0m"
    LOG_TEXT: str = "\033[37m"  # WHITE

    def __init__(self):
        self.client = None

    def log(self, message: str):
        text = self.LOG_COLOR + self.LOG_TEXT + f"[{self.NAME}] " + message + self.LOG_RESET
        logging.info(text)

    def init_client_as_needed(self):
        """Lazy-init the OpenAI client. Subclasses call this before first LLM call."""
        if not self.client:
            from openai import OpenAI

            from src.openai_key import get_openai_api_key

            self.log("Initialising OpenAI client")
            self.client = OpenAI(api_key=get_openai_api_key())
            self.log("OpenAI client ready")

    @abstractmethod
    async def run(
        self,
        classifier_output: ClassifierOutput,
        user_profile: Optional[UserProfile] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator. Yield text chunks that will be streamed to the client.
        Must never raise — catch internally and yield an error message instead.
        """
        ...
