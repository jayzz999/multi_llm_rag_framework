"""
OpenAI LLM Implementation
GPT-4, GPT-3.5-turbo, etc.
"""

import os
from typing import Iterator, Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseLLM, LLMResponse, LLMProvider

load_dotenv()


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM implementation.
    
    Supports:
    - gpt-4 (most capable)
    - gpt-4-turbo
    - gpt-3.5-turbo (cheaper, faster)
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            api_key: OpenAI API key (if not in environment)
            **kwargs: Additional parameters
            
        Example:
            # GPT-3.5 (cheaper)
            llm = OpenAILLM(model="gpt-3.5-turbo")
            
            # GPT-4 (better quality)
            llm = OpenAILLM(model="gpt-4")
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai tiktoken"
            )
        
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        self.provider = LLMProvider.OPENAI
        
        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print(f"✓ OpenAI LLM initialized: {model}")
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from OpenAI.
        
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
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() 
                   if k not in ["temperature", "max_tokens"]}
            )
            
            # Extract content
            content = response.choices[0].message.content
            
            # Create response object
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "provider": "openai",
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
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
        Generate a streaming response from OpenAI.
        
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
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Stream response
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            raise Exception(f"Error streaming response: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def get_cost_estimate(
        self, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> float:
        """
        Estimate cost for given token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
        }
        
        # Get base model name
        base_model = self.model.split('-')[0:2]
        base_model = '-'.join(base_model)
        
        prices = pricing.get(base_model, {"input": 0.5, "output": 1.5})
        
        input_cost = (prompt_tokens / 1_000_000) * prices["input"]
        output_cost = (completion_tokens / 1_000_000) * prices["output"]
        
        return input_cost + output_cost