"""
Performance Lab v2.0 - LiteLLM Configuration
===========================================
Unified LLM interface for ComfyUI workflow optimization.

Supports multiple providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models (Ollama, LM Studio)
- Open source (via Together AI, Replicate)

Features:
- Automatic fallback on errors
- Cost tracking per request
- Performance metrics
- Quality tier routing
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
from enum import Enum

# LiteLLM will be imported lazily to avoid hard dependency
# pip install litellm

logger = logging.getLogger(__name__)


# ============================================================================
# QUALITY TIERS
# ============================================================================

class QualityTier(Enum):
    """Quality tiers for LLM selection."""
    CRITICAL = "critical"    # Best available model, highest cost
    IMPORTANT = "important"  # High quality, balanced cost
    STANDARD = "standard"    # Default tier, good quality
    QUICK = "quick"          # Fast local model, low/no cost


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    # Critical tier - best models
    QualityTier.CRITICAL: [
        {
            "name": "gpt-4-turbo-preview",
            "provider": "openai",
            "max_tokens": 4096,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.01, "output": 0.03},
            "latency_estimate": "high"
        },
        {
            "name": "claude-3-opus-20240229",
            "provider": "anthropic",
            "max_tokens": 4096,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.015, "output": 0.075},
            "latency_estimate": "high"
        }
    ],
    
    # Important tier - balanced quality/cost
    QualityTier.IMPORTANT: [
        {
            "name": "gpt-4-turbo",
            "provider": "openai",
            "max_tokens": 2048,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.01, "output": 0.03},
            "latency_estimate": "medium"
        },
        {
            "name": "claude-3-sonnet-20240229",
            "provider": "anthropic",
            "max_tokens": 2048,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
            "latency_estimate": "medium"
        }
    ],
    
    # Standard tier - good quality, lower cost
    QualityTier.STANDARD: [
        {
            "name": "gpt-3.5-turbo",
            "provider": "openai",
            "max_tokens": 2048,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015},
            "latency_estimate": "low"
        },
        {
            "name": "claude-3-haiku-20240307",
            "provider": "anthropic",
            "max_tokens": 2048,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.00025, "output": 0.00125},
            "latency_estimate": "low"
        }
    ],
    
    # Quick tier - fast local models
    QualityTier.QUICK: [
        {
            "name": "ollama/llama3.1:8b",
            "provider": "ollama",
            "max_tokens": 1024,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.0, "output": 0.0},
            "latency_estimate": "very_low"
        },
        {
            "name": "ollama/mistral:7b",
            "provider": "ollama",
            "max_tokens": 1024,
            "temperature": 0.7,
            "cost_per_1k_tokens": {"input": 0.0, "output": 0.0},
            "latency_estimate": "very_low"
        }
    ]
}


# ============================================================================
# LITELLM MANAGER
# ============================================================================

class LiteLLMManager:
    """
    Manages LLM requests across multiple providers.
    
    Features:
    - Quality tier routing
    - Automatic fallback on errors
    - Cost tracking
    - Performance metrics
    - Request caching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.request_history: List[Dict[str, Any]] = []
        self.total_cost = 0.0
        self.total_requests = 0
        self.failed_requests = 0
        
        # API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Lazy import litellm
        try:
            import litellm
            self.litellm = litellm
            
            # Configure litellm
            if self.openai_api_key:
                litellm.openai_key = self.openai_api_key
            if self.anthropic_api_key:
                litellm.anthropic_key = self.anthropic_api_key
                
            # Suppress excessive logs
            litellm.suppress_debug_info = True
            
            logger.info("LiteLLM initialized successfully")
            
        except ImportError:
            logger.warning("litellm not installed. Install with: pip install litellm")
            self.litellm = None
            
    async def generate(
        self,
        prompt: str,
        quality_tier: QualityTier = QualityTier.STANDARD,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using quality tier-appropriate model.
        
        Args:
            prompt: User prompt
            quality_tier: Quality tier for model selection
            max_tokens: Optional max tokens override
            temperature: Optional temperature override
            system_prompt: Optional system prompt
            user_metadata: Optional metadata for tracking
            
        Returns:
            Dict with generated text and metadata
        """
        if not self.litellm:
            return {
                "success": False,
                "error": "LiteLLM not available",
                "text": ""
            }
        
        # Get models for quality tier
        models = MODEL_CONFIGS.get(quality_tier, MODEL_CONFIGS[QualityTier.STANDARD])
        
        # Try each model with fallback
        last_error = None
        for model_config in models:
            try:
                result = await self._try_model(
                    model_config=model_config,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                # Track successful request
                self._track_request(
                    model_config=model_config,
                    prompt=prompt,
                    result=result,
                    success=True,
                    user_metadata=user_metadata
                )
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_config['name']} failed: {e}, trying fallback...")
                continue
        
        # All models failed
        self.failed_requests += 1
        logger.error(f"All models failed for quality tier {quality_tier.value}")
        
        return {
            "success": False,
            "error": f"All models failed: {last_error}",
            "text": ""
        }
    
    async def _try_model(
        self,
        model_config: Dict[str, Any],
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Try to generate with a specific model."""
        model_name = model_config["name"]
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use config defaults if not overridden
        final_max_tokens = max_tokens or model_config["max_tokens"]
        final_temperature = temperature if temperature is not None else model_config["temperature"]
        
        # Call LiteLLM
        logger.info(f"Calling {model_name} (max_tokens={final_max_tokens}, temp={final_temperature})")
        
        response = await self.litellm.acompletion(
            model=model_name,
            messages=messages,
            max_tokens=final_max_tokens,
            temperature=final_temperature
        )
        
        # Extract response
        generated_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = self._calculate_cost(model_config, input_tokens, output_tokens)
        
        return {
            "success": True,
            "text": generated_text,
            "model": model_name,
            "tokens_used": tokens_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_cost(
        self,
        model_config: Dict[str, Any],
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for request."""
        cost_config = model_config["cost_per_1k_tokens"]
        
        input_cost = (input_tokens / 1000) * cost_config["input"]
        output_cost = (output_tokens / 1000) * cost_config["output"]
        
        total_cost = input_cost + output_cost
        self.total_cost += total_cost
        
        return total_cost
    
    def _track_request(
        self,
        model_config: Dict[str, Any],
        prompt: str,
        result: Dict[str, Any],
        success: bool,
        user_metadata: Optional[Dict[str, Any]]
    ):
        """Track request for analytics."""
        self.total_requests += 1
        
        request_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_config["name"],
            "provider": model_config["provider"],
            "success": success,
            "prompt_length": len(prompt),
            "tokens_used": result.get("tokens_used", 0),
            "cost": result.get("cost", 0.0),
            "metadata": user_metadata or {}
        }
        
        self.request_history.append(request_data)
        
        # Keep last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{success_rate:.1f}%",
            "total_cost": f"${self.total_cost:.4f}",
            "average_cost_per_request": (
                f"${self.total_cost / self.total_requests:.4f}"
                if self.total_requests > 0 else "$0.0000"
            )
        }
    
    def get_recent_requests(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request history."""
        return self.request_history[-limit:]


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Global manager instance
litellm_manager = LiteLLMManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def generate_critical(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate with CRITICAL quality tier."""
    return await litellm_manager.generate(prompt, QualityTier.CRITICAL, **kwargs)


async def generate_important(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate with IMPORTANT quality tier."""
    return await litellm_manager.generate(prompt, QualityTier.IMPORTANT, **kwargs)


async def generate_standard(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate with STANDARD quality tier."""
    return await litellm_manager.generate(prompt, QualityTier.STANDARD, **kwargs)


async def generate_quick(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate with QUICK quality tier."""
    return await litellm_manager.generate(prompt, QualityTier.QUICK, **kwargs)
