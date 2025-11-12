"""
Ollama LLM Implementation
FREE local LLMs - Llama 3.1, Mistral, etc.
"""

import os
from typing import Iterator, Optional
from dotenv import load_dotenv

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .base import BaseLLM, LLMResponse, LLMProvider

load_dotenv()


class OllamaLLM(BaseLLM):
    """
    Ollama LLM implementation.
    
    Runs LLMs locally - 100% FREE!
    
    Popular models:
    - llama3.1 (8B, 70B, 405B parameters)
    - mistral (7B parameters)
    - mixtral (8x7B parameters)
    - codellama (for coding)
    
    Installation:
    1. Install Ollama: https://ollama.ai
    2. Pull model: ollama pull llama3.1
    """
    
    def __init__(
        self,
        model: str = "llama3.1",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model: Model name (llama3.1, mistral, mixtral, etc.)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            base_url: Ollama server URL (default: http://localhost:11434)
            **kwargs: Additional parameters
            
        Example:
            # Use Llama 3.1
            llm = OllamaLLM(model="llama3.1", temperature=0.7)
            
            # Use Mistral
            llm = OllamaLLM(model="mistral", temperature=0.5)
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama package not installed. "
                "Install with: pip install ollama"
            )
        
        super().__init__(model, temperature, max_tokens, **kwargs)
        
        self.provider = LLMProvider.OLLAMA
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", 
            "http://localhost:11434"
        )
        
        # Check if model is available
        self._check_model_availability()
        
        print(f"✓ Ollama LLM initialized: {model}")
    
    def _check_model_availability(self) -> None:
        """Check if the model is available locally."""
        try:
            # List available models
            models = ollama.list()
            available_models = [m['name'].split(':')[0] for m in models['models']]
            
            if self.model not in available_models:
                print(f"⚠ Warning: Model '{self.model}' not found locally.")
                print(f"  Available models: {', '.join(available_models)}")
                print(f"  To download: ollama pull {self.model}")
        
        except Exception as e:
            print(f"⚠ Could not check model availability: {str(e)}")
            print(f"  Make sure Ollama is running!")
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from Ollama.
        
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
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            )
            
            # Extract content
            content = response['message']['content']
            
            # Create response object
            llm_response = LLMResponse(
                content=content,
                model=self.model,
                finish_reason="stop",
                metadata={
                    "provider": "ollama",
                    "base_url": self.base_url
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
        Generate a streaming response from Ollama.
        
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
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        
        except Exception as e:
            raise Exception(f"Error streaming response: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (approximate for Ollama models).
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate number of tokens
        """
        # Rough estimation: ~4 characters per token
        # This is approximate - exact tokenization varies by model
        return len(text) // 4
    
    def get_available_models(self) -> list:
        """
        Get list of available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            models = ollama.list()
            return [m['name'] for m in models['models']]
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []