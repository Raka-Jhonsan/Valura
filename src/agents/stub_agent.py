"""
Stub Agent — handles all agents not yet implemented in this build.

Returns a structured, informative response that includes:
  - The classified intent
  - Extracted entities
  - Which agent would have handled this
  - A clear message that this agent is not yet implemented

Never crashes. Never returns a raw error. The router always has somewhere to go.
"""
import json
from typing import AsyncGenerator, Optional
from src.models import ClassifierOutput, UserProfile, StubAgentResponse
from src.agents.base_agent import BaseAgent

# Colors
BG_BLACK = "\033[40m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class StubAgent(BaseAgent):
    """
    Placeholder for agents not yet implemented.
    Receives the classifier output and streams a structured acknowledgement.
    """

    NAME = "Stub Agent"
    DESCRIPTION = "Placeholder for unimplemented agents"
    LOG_COLOR = BG_BLACK
    LOG_TEXT = YELLOW
    LOG_RESET = RESET

    def __init__(self):
        super().__init__()

    async def run(
        self,
        classifier_output: ClassifierOutput,
        user_profile: Optional[UserProfile] = None,
    ) -> AsyncGenerator[str, None]:
        self.log(f"Stub handling agent: {classifier_output.target_agent}")

        response = StubAgentResponse(
            classified_intent=classifier_output.intent,
            extracted_entities=classifier_output.entities,
            target_agent=classifier_output.target_agent,
            message=(
                f"The '{classifier_output.target_agent}' agent is not yet implemented in this build. "
                f"Your query was correctly classified as '{classifier_output.intent}' and would be "
                f"handled by the {classifier_output.target_agent} specialist. "
                f"This capability is coming soon."
            ),
        )

        yield json.dumps({"type": "stub", "data": response.model_dump()}) + "\n"
        yield json.dumps(
            {
                "type": "text",
                "data": response.message,
            }
        ) + "\n"
