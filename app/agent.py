import json
import logging
import re
from typing import List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.catalog import CatalogSearch
from app.models import Message, Recommendation, ChatResponse
from app.prompts import build_system_prompt

logger = logging.getLogger(__name__)

MAX_RECOMMENDATIONS = 10


def _build_search_query(messages: List[Message]) -> str:
    """
    Build a semantic search query from the conversation.
    Concatenates all user messages (last 4) so FAISS retrieves
    the most contextually relevant catalogue items.
    """
    user_msgs = [m.content for m in messages if m.role == "user"]
    return " ".join(user_msgs[-4:])


def _extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from the LLM's text output.
    Handles cases where the model wraps in markdown fences or adds preamble.
    """
    # 1. Try direct parse (JSON mode should give us clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _parse_response(
    text: str,
    valid_urls: set,
) -> Tuple[str, Optional[List[Recommendation]], bool]:
    """
    Parse the LLM response into (reply, recommendations, end_of_conversation).
    Falls back gracefully if JSON parsing fails.
    """
    data = _extract_json(text)

    if data is None:
        # Couldn't parse JSON — return the raw text as the reply
        logger.warning("Could not parse JSON from LLM response; returning raw text.")
        return text, None, False

    reply = data.get("reply", "")
    if not reply:
        # Sometimes the model puts everything in reply; use full text as fallback
        reply = text

    eoc = bool(data.get("end_of_conversation", False))

    recs_raw = data.get("recommendations")
    recommendations: Optional[List[Recommendation]] = None

    if recs_raw and isinstance(recs_raw, list) and len(recs_raw) > 0:
        validated = []
        for r in recs_raw[:MAX_RECOMMENDATIONS]:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "")
            name = r.get("name", "")
            test_type = r.get("test_type", "")
            if not name or not url:
                continue
            # Soft URL validation: must start with the SHL domain
            if not url.startswith("https://www.shl.com/"):
                logger.warning(f"Non-SHL URL rejected: {url}")
                continue
            validated.append(Recommendation(name=name, url=url, test_type=test_type))
        if validated:
            recommendations = validated

    return reply, recommendations, eoc


class SHLAgent:
    def __init__(self):
        self.catalog_search = CatalogSearch()
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1500,
        ).bind(response_format={"type": "json_object"})

        # Pre-compute valid URLs for validation
        self._valid_urls = {item["url"] for item in self.catalog_search.get_all()}

    async def chat(self, messages: List[Message]) -> ChatResponse:
        """Main entry point: takes full conversation history, returns next agent reply."""

        # --- 1. Semantic retrieval ---
        query = _build_search_query(messages)
        relevant_items = self.catalog_search.search(query, k=20)

        # --- 2. Build system prompt with grounded catalogue context ---
        system_prompt = build_system_prompt(relevant_items)

        # --- 3. Assemble LangChain message list ---
        lc_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))

        # --- 4. Call the LLM ---
        try:
            response = await self.llm.ainvoke(lc_messages)
            raw_text = response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ChatResponse(
                reply="I'm sorry, I encountered an error processing your request. Please try again.",
                recommendations=None,
                end_of_conversation=False,
            )

        # --- 5. Parse and validate ---
        reply, recommendations, eoc = _parse_response(raw_text, self._valid_urls)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=eoc,
        )
