"""Importance scoring service"""

import logging
from typing import Dict

from sekha_llm_bridge.utils.llm_client import llm_client
from sekha_llm_bridge.models.requests import ScoreRequest
from sekha_llm_bridge.models.responses import ScoreResponse

logger = logging.getLogger(__name__)


class ImportanceScorerService:
    """Service for scoring conversation importance"""
    
    SCORING_PROMPT = """Rate the importance of the following conversation on a scale of 1-10, where:
1-3: Casual, trivial, or redundant information
4-6: Useful information, minor decisions
7-9: Important discussions, key decisions, valuable insights
10: Critical information, major decisions, breakthrough insights

Consider:
- Decision-making content
- Novel information
- Action items
- Long-term relevance

Conversation:
{conversation}

Provide a score (1-10) and brief reasoning:"""
    
    async def score_importance(self, request: ScoreRequest) -> ScoreResponse:
        """Score conversation importance"""
        logger.info(f"Scoring importance for conversation (length: {len(request.conversation)})")
        
        system_msg = "You are an expert at evaluating conversation importance. Be objective and concise."
        user_msg = self.SCORING_PROMPT.format(conversation=request.conversation)
        
        try:
            response = await llm_client.generate_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                model=request.model,
                temperature=0.3,
                max_tokens=200
            )
            
            # Extract score from response
            score = self._extract_score(response)
            
            return ScoreResponse(
                score=score,
                reasoning=response.strip(),
                model=request.model or "default"
            )
        
        except Exception as e:
            logger.error(f"Importance scoring failed: {e}")
            # Return default score on failure
            return ScoreResponse(
                score=5.0,
                reasoning="Failed to score importance",
                model=request.model or "default"
            )
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from LLM response"""
        import re
        
        # Look for patterns like "Score: 7" or "7/10" or just "7"
        patterns = [
            r'score:?\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)/10',
            r'^(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 1.0), 10.0)  # Clamp to 1-10
                except ValueError:
                    continue
        
        # Default to 5 if no score found
        return 5.0


importance_scorer_service = ImportanceScorerService()
