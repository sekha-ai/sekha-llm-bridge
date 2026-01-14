"""Summarization service"""

import logging

from sekha_llm_bridge.utils.llm_client import llm_client
from sekha_llm_bridge.models.requests import SummarizeRequest
from sekha_llm_bridge.models.responses import SummarizeResponse

logger = logging.getLogger(__name__)


class SummarizationService:
    """Service for generating summaries"""
    
    PROMPTS = {
        "daily": """Summarize the following conversation from today. Focus on key topics, decisions, and action items. Be concise but comprehensive.

Conversation:
{messages}

Summary:""",
        
        "weekly": """Synthesize the following daily summaries from this week into a cohesive weekly summary. Identify recurring themes, progress on topics, and overall trends.

Daily Summaries:
{messages}

Weekly Summary:""",
        
        "monthly": """Create a high-level monthly summary from the following weekly summaries. Focus on major themes, significant developments, and long-term patterns.

Weekly Summaries:
{messages}

Monthly Summary:"""
    }
    
    async def generate_summary(self, request: SummarizeRequest) -> SummarizeResponse:
        """Generate summary from messages"""
        logger.info(f"Generating {request.level} summary from {len(request.messages)} messages")
        
        # Get appropriate prompt
        prompt_template = self.PROMPTS.get(request.level, self.PROMPTS["daily"])
        
        # Format messages
        messages_text = "\n\n".join([
            f"[{i+1}] {msg}" for i, msg in enumerate(request.messages)
        ])
        
        # Build LLM prompt
        system_msg = "You are a helpful assistant that creates concise, informative summaries."
        user_msg = prompt_template.format(messages=messages_text)
        
        try:
            summary = await llm_client.generate_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                model=request.model,
                temperature=0.3,  # Low temperature for consistent summaries
                max_tokens=500
            )
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            input_tokens = len(messages_text) // 4
            output_tokens = len(summary) // 4
            tokens_used = input_tokens + output_tokens
            
            return SummarizeResponse(
                summary=summary.strip(),
                level=request.level,
                model=request.model or "default",
                message_count=len(request.messages),
                tokens_used=tokens_used  # ADDED: Required field
            )
        
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise


summarization_service = SummarizationService()