"""
Anthropic Claude LLM Implementation
Claude 3.5 Sonnet, Claude 3 Opus, etc.
"""

import os
from typing import Iterator, Optional
from dotenv import load_dotenv

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import BaseLLM, LLMResponse, LLMProvider

load_dotenv()


class ClaudeLLM(BaseLLM):
    """
    Anthropic Claude LLM implementation.
    
    Supports:
    - claude-3-5-sonnet-20241022 (best balance)
    - claude-3-opus-20240229 (most capable)
    - claude-3-sonnet-20240229
    - claude-3-haiku-20240307 (fastest)
    """
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Claude LLM.
        
        Args:
            model: Model name
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            api_key: Anthropic API key (if not in environment)
            **kwargs: Additional parameters
            
        Example:
            # Claude 3.5 Sonnet (recommended)
            llm = ClaudeLLM(model="claude-3-5-sonnet-20241022")
            
            # Claude 3 Opus (most capable)
            llm = ClaudeLLM(model="claude-3-opus-20240229")
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        self.provider = LLMProvider.ANTHROPIC
        
        # Get API key
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize client
        self.client = Anthropic(api_key=self.api_key)
        
        print(f"✓ Claude LLM initialized: {model}")
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from Claude.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt for context
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse object with generated content
            
        Example:
            response = llm.generate("What is RAG?")
            print(response.content)
        """
        try:
            # Prepare system prompt
            system = system_prompt or "You are a helpful AI assistant."
            
            # Generate response
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract content
            content = response.content[0].text
            
            # Create response object
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                metadata={
                    "provider": "anthropic",
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
            return llm_response
        
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def generate_stream(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a streaming response from Claude.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt for context
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of generated text
            
        Example:
            for chunk in llm.generate_stream("Explain AI"):
                print(chunk, end="", flush=True)
        """
        try:
            # Prepare system prompt
            system = system_prompt or "You are a helpful AI assistant."
            
            # Stream response
            with self.client.messages.stream(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        
        except Exception as e:
            raise Exception(f"Error streaming response: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (approximate for Claude).
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        # Anthropic uses ~3.5 characters per token on average
        return len(text) // 4
    
    def get_cost_estimate(
        self, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """
        Estimate cost for given token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25}
        }
        
        # Get base model name
        for model_key in pricing.keys():
            if model_key in self.model:
                prices = pricing[model_key]
                break
        else:
            prices = {"input": 3.0, "output": 15.0}
        
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        
        return input_cost + output_cost