import os
import re

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.core import settings, get_logger
from src.modules.agents.state import AgentState


logger = get_logger(module="agents", node="analyze")


def _normalize_ollama_base_url(url: str) -> str:
    clean = url.rstrip("/")
    if clean.endswith("/v1"):
        return clean
    return f"{clean}/v1"


class AnalystResponse(BaseModel):
    draft_answer: str = Field(min_length=1)
    needs_more_context: bool = False


class AnalystNode:
    def __init__(self) -> None:
        self._agent: Agent[None, AnalystResponse] | None = None

    @staticmethod
    def _provider_model() -> str:
        if settings.llm.provider == "gemini":
            if settings.llm.api_key is not None and "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = settings.llm.api_key.get_secret_value()
            return f"google-gla:{settings.llm.model_name}"

        if settings.llm.provider == "ollama":
            if "OLLAMA_BASE_URL" not in os.environ:
                os.environ["OLLAMA_BASE_URL"] = _normalize_ollama_base_url(settings.llm.ollama_host)
            return f"ollama:{settings.llm.model_name}"

        return settings.llm.model_name

    def _get_agent(self) -> Agent[None, AnalystResponse]:
        if self._agent is None:
            model_name = self._provider_model()
            logger.info("Initializing analyst agent with model={}", model_name)
            self._agent = Agent(
                model=model_name,
                output_type=AnalystResponse,
                system_prompt=(
                    "You are an analyst in an ARCADE pipeline. "
                    "Use only the provided context. "
                    "If context is insufficient, set needs_more_context=true and explain constraints."
                ),
            )
        return self._agent

    @staticmethod
    def _synthesize_fallback_answer(question: str, context: str) -> tuple[str, bool]:
        if not context.strip():
            return "Insufficient context found in knowledge base for this question.", True

        # Strip common LightRAG wrappers to avoid returning raw technical dumps.
        cleaned = re.sub(r"```(?:json)?", "", context, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        cleaned = re.sub(r"Knowledge Graph Data \([^\n]+\):", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Vector Data \([^\n]+\):", "", cleaned, flags=re.IGNORECASE)

        desc_pattern = r'["“]description["”]\s*:\s*["“]([^"”]+)["”]'
        descriptions = [d.strip() for d in re.findall(desc_pattern, cleaned) if d.strip()]

        if descriptions:
            question_tokens = set(re.findall(r"[a-zA-Zа-яА-Я0-9]{4,}", question.lower()))

            def score(text: str) -> int:
                text_tokens = set(re.findall(r"[a-zA-Zа-яА-Я0-9]{4,}", text.lower()))
                return len(question_tokens & text_tokens)

            ranked = sorted(descriptions, key=score, reverse=True)
            top = []
            for item in ranked:
                if item not in top:
                    top.append(item)
                if len(top) == 2:
                    break
            return " ".join(top), False

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        if sentences:
            return " ".join(sentences[:2]), False

        return "I found context, but could not extract a reliable answer from it.", True

    async def __call__(self, state: AgentState) -> AgentState:
        logger.info(
            "Analyze node started: iteration={}, context_len={}",
            state.get("iteration", 0),
            len(state.get("context", "")),
        )
        prompt = (
            f"Question:\n{state['question']}\n\n"
            f"Context:\n{state.get('context', '')}\n\n"
            f"Critique:\n{state.get('critique', '')}\n\n"
            "Return concise, factual answer with explicit uncertainty when needed."
        )

        try:
            result = await self._get_agent().run(prompt)
            data = result.output
            draft_answer = data.draft_answer
            needs_more_context = data.needs_more_context
            logger.info(
                "Analyze node completed: draft_len={}, needs_more_context={}",
                len(draft_answer),
                needs_more_context,
            )
        except Exception:
            # Fallback keeps pipeline resilient when provider is misconfigured.
            logger.exception("Analyze node failed, using fallback draft generation")
            draft_answer, needs_more_context = self._synthesize_fallback_answer(
                question=state["question"],
                context=state.get("context", ""),
            )
            logger.warning(
                "Analyze fallback produced draft: draft_len={}, needs_more_context={}",
                len(draft_answer),
                needs_more_context,
            )

        events = list(state.get("events", []))
        events.append(
            {
                "type": "analyze_completed",
                "data": "Analyst produced a draft answer",
                "iteration": state.get("iteration", 0),
            }
        )

        return {
            **state,
            "draft_answer": draft_answer,
            "needs_more_context": needs_more_context,
            "events": events,
        }
