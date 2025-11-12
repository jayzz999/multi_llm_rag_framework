"""
Base class for Large Language Models
All LLM implementations must inherit from this
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """
    Represents a response from an LLM.
    """
    content: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __repr__(self) -> str:
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"LLMResponse(content='{preview}', model='{self.model}')"


class BaseLLM(ABC):
    """
    Abstract base class for LLMs.
    
    All LLM implementations (OpenAI, Claude, Ollama)
    must inherit from this class and implement the required methods.
    """
    
    def __init__(
        self, 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize the LLM.
        
        Args:
            model: Model name/identifier
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = None  # Set by implementations
        self.additional_params = kwargs
    
    @abstractmethod
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
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
        pass
    
    @abstractmethod
    def generate_stream(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a streaming response from the LLM.
        
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
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
            
        Example:
            tokens = llm.count_tokens("Hello world")
            # Returns: 2
        """
        pass
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with retrieved context (for RAG).
        
        Args:
            query: User query
            context: List of context strings (retrieved documents)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
            
        Example:
            context = ["RAG is...", "Vector databases are..."]
            response = llm.generate_with_context("What is RAG?", context)
        """
        # Construct prompt with context
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" 
                                    for i, ctx in enumerate(context)])
        
        full_prompt = f"""Based on the following context, answer the question.

{context_text}

Question: {query}

Answer:"""
        
        return self.generate(full_prompt, system_prompt=system_prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "provider": self.provider.value if self.provider else "unknown",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def __repr__(self) -> str:
        """String representation of the LLM."""
        return (f"{self.__class__.__name__}(model='{self.model}', "
                f"temperature={self.temperature})")
