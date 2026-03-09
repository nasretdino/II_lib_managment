import os

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.core import settings, get_logger
from src.modules.agents.state import AgentState


logger = get_logger(module="agents", node="critique")


class CriticResponse(BaseModel):
    is_approved: bool
    critique: str = Field(default="")
    citations_valid: bool = True
    needs_more_context: bool = False


class CriticNode:
    def __init__(self) -> None:
        self._agent: Agent[None, CriticResponse] | None = None

    @staticmethod
    def _provider_model() -> str:
        if settings.llm.provider == "gemini":
            if settings.llm.api_key is not None and "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = settings.llm.api_key.get_secret_value()
            return f"google-gla:{settings.llm.model_name}"

        if settings.llm.provider == "ollama":
            return f"ollama:{settings.llm.model_name}"

        return settings.llm.model_name

    def _get_agent(self) -> Agent[None, CriticResponse]:
        if self._agent is None:
            model_name = self._provider_model()
            logger.info("Initializing critic agent with model={}", model_name)
            self._agent = Agent(
                model=model_name,
                result_type=CriticResponse,
                system_prompt=(
                    "You are a strict critic in ARCADE reflection loop. "
                    "Do not rewrite answer. Only evaluate factual grounding against context, "
                    "consistency, and citation validity."
                ),
            )
        return self._agent

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info(
            "Critique node started: iteration={}, draft_len={}, context_len={}",
            state.get("iteration", 0),
            len(state.get("draft_answer", "")),
            len(state.get("context", "")),
        )
        prompt = (
            f"Question:\n{state['question']}\n\n"
            f"Context:\n{state.get('context', '')}\n\n"
            f"Draft answer:\n{state.get('draft_answer', '')}\n\n"
            "Return is_approved=true only if answer is grounded in context and internally consistent."
        )

        try:
            result = await self._get_agent().run(prompt)
            data = result.output
            is_approved = data.is_approved
            critique = data.critique
            citations_valid = data.citations_valid
            needs_more_context = data.needs_more_context
            logger.info(
                "Critique node completed: approved={}, citations_valid={}, needs_more_context={}",
                is_approved,
                citations_valid,
                needs_more_context,
            )
        except Exception:
            logger.exception("Critique node failed, using fallback evaluator")
            draft = state.get("draft_answer", "")
            context = state.get("context", "")
            is_approved = bool(draft and context)
            critique = "Automatic fallback critic used."
            citations_valid = True
            needs_more_context = not bool(context)
            logger.warning(
                "Critique fallback result: approved={}, needs_more_context={}",
                is_approved,
                needs_more_context,
            )

        events = list(state.get("events", []))
        events.append(
            {
                "type": "critique_completed",
                "data": critique or "Critic approved the draft",
                "iteration": state.get("iteration", 0),
            }
        )

        return {
            **state,
            "is_approved": is_approved,
            "critique": critique,
            "citations_valid": citations_valid,
            "needs_more_context": needs_more_context,
            "events": events,
        }
