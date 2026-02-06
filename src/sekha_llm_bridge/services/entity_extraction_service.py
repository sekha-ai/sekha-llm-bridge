"""Entity extraction service"""

import json
import logging

from sekha_llm_bridge.models.requests import ExtractRequest
from sekha_llm_bridge.models.responses import ExtractResponse
from sekha_llm_bridge.utils.llm_client import llm_client

logger = logging.getLogger(__name__)


class EntityExtractionService:
    """Service for extracting entities from text"""

    EXTRACTION_PROMPT = """Extract key entities from the following text. Identify:
- People (names, roles)
- Organizations (companies, teams)
- Technical terms (technologies, frameworks, tools)
- Concepts (ideas, methodologies)

Text:
{text}

Return a JSON object with arrays for each entity type:
{{
"people": ["name1", "name2"],
"organizations": ["org1"],
"technical_terms": ["term1", "term2"],
"concepts": ["concept1"]
}}
"""

    async def extract_entities(self, request: ExtractRequest) -> ExtractResponse:
        """Extract entities from text"""
        logger.info(f"Extracting entities from text (length: {len(request.text)})")

        system_msg = "You are an expert at extracting and categorizing entities from text. Always return valid JSON."
        user_msg = self.EXTRACTION_PROMPT.format(text=request.text)

        try:
            response = await llm_client.generate_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                model=request.model,
                temperature=0.2,
                max_tokens=1000,
            )

            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            entities = json.loads(response.strip())

            return ExtractResponse(entities=entities, model=request.model or "default")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity extraction JSON: {e}")
            return ExtractResponse(
                entities={
                    "people": [],
                    "organizations": [],
                    "technical_terms": [],
                    "concepts": [],
                },
                model=request.model or "default",
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise


entity_extraction_service = EntityExtractionService()
