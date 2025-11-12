"""
Google Gemini LLM Implementation
Supports Gemini Pro and other Gemini models
"""

import os
from typing import Iterator, Optional
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .base import BaseLLM, LLMResponse, LLMProvider

load_dotenv()


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM implementation.

    Supports:
    - gemini-pro (best for text)
    - gemini-pro-vision (for multimodal)
    """

    def __init__(
        self,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemini LLM.

        Args:
            model: Model name
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            api_key: Google API key (if not in environment)
            **kwargs: Additional parameters

        Example:
            llm = GeminiLLM(model="gemini-pro")
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )

        super().__init__(model, temperature, max_tokens, **kwargs)

        self.provider = LLMProvider.GEMINI if hasattr(LLMProvider, 'GEMINI') else None

        # Get API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.client = genai.GenerativeModel(model)

        print(f"✓ Gemini LLM initialized: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from Gemini.

        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt (not used by Gemini)
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object with generated content
        """
        try:
            # Combine system prompt with user prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Generate response
            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
            )

            # Extract content
            content = response.text

            # Create response object
            llm_response = LLMResponse(
                content=content,
                model=self.model,
                tokens_used=None,  # Gemini doesn't provide token count in response
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                metadata={
                    "provider": "gemini",
                    "safety_ratings": [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name
                        }
                        for rating in response.candidates[0].safety_ratings
                    ] if response.candidates else []
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
        Generate a streaming response from Gemini.

        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Chunks of generated text
        """
        try:
            # Combine system prompt with user prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Stream response
            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                ),
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            raise Exception(f"Error streaming response: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Gemini's tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            result = self.client.count_tokens(text)
            return result.total_tokens
        except:
            # Fallback to character-based estimate
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
        # Gemini Pro pricing (as of 2024)
        # Free tier: 60 requests per minute
        # Paid: $0.00025 per 1K characters (~$0.001 per 1K tokens)

        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1_000) * 0.001

        return cost
