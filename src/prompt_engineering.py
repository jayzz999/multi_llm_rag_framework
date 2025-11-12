"""
Prompt Engineering
Templates and strategies for effective prompting
"""

from typing import List, Dict, Any, Optional
from enum import Enum


class PromptStrategy(Enum):
    """Different prompting strategies."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    CUSTOM = "custom"


class PromptTemplate:
    """
    Prompt template management for RAG systems.
    
    Implements various prompting strategies:
    - Zero-shot: Direct question answering
    - Few-shot: Learning from examples
    - Chain-of-thought: Step-by-step reasoning
    - ReAct: Reasoning + Acting
    """
    
    # ========================================
    # SYSTEM PROMPTS
    # ========================================
    
    SYSTEM_PROMPTS = {
        "default": """You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context.""",
        
        "detailed": """You are an expert AI assistant with deep knowledge across multiple domains. 
Provide comprehensive, well-structured answers based on the given context.
If the context doesn't contain sufficient information, clearly state what's missing.""",
        
        "concise": """You are a precise AI assistant. Provide brief, direct answers based on the context. 
Keep responses under 3 sentences unless more detail is specifically requested.""",
        
        "academic": """You are an academic research assistant. Provide scholarly, well-reasoned responses.
Cite specific sources when using information. Maintain objectivity and acknowledge limitations.""",
        
        "creative": """You are a creative and insightful AI assistant. Provide engaging, thoughtful responses.
Feel free to draw connections and offer unique perspectives while staying grounded in the context."""
    }
    
    # ========================================
    # ZERO-SHOT PROMPTS
    # ========================================
    
    @staticmethod
    def zero_shot(
        context: str,
        question: str,
        instructions: Optional[str] = None
    ) -> str:
        """
        Zero-shot prompting - direct question answering.
        
        Args:
            context: Retrieved context documents
            question: User question
            instructions: Optional additional instructions
            
        Returns:
            Formatted prompt
            
        Example:
            prompt = PromptTemplate.zero_shot(
                context="RAG is...",
                question="What is RAG?"
            )
        """
        base_instructions = instructions or "Answer the question based on the context below."
        
        prompt = f"""{base_instructions}

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    # ========================================
    # FEW-SHOT PROMPTS
    # ========================================
    
    @staticmethod
    def few_shot(
        context: str,
        question: str,
        examples: List[Dict[str, str]],
        instructions: Optional[str] = None
    ) -> str:
        """
        Few-shot prompting - learning from examples.
        
        Args:
            context: Retrieved context documents
            question: User question
            examples: List of example Q&A pairs [{"question": "...", "answer": "..."}]
            instructions: Optional additional instructions
            
        Returns:
            Formatted prompt
            
        Example:
            examples = [
                {"question": "What is ML?", "answer": "Machine Learning is..."},
                {"question": "What is AI?", "answer": "Artificial Intelligence is..."}
            ]
            prompt = PromptTemplate.few_shot(
                context="...",
                question="What is RAG?",
                examples=examples
            )
        """
        base_instructions = instructions or "Answer questions based on the context, following the example format."
        
        # Build examples section
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"""Example {i}:
Question: {example['question']}
Answer: {example['answer']}

"""
        
        prompt = f"""{base_instructions}

{examples_text}Now answer the following question based on the context:

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    # ========================================
    # CHAIN-OF-THOUGHT PROMPTS
    # ========================================
    
    @staticmethod
    def chain_of_thought(
        context: str,
        question: str,
        include_reasoning: bool = True
    ) -> str:
        """
        Chain-of-thought prompting - step-by-step reasoning.
        
        Args:
            context: Retrieved context documents
            question: User question
            include_reasoning: Whether to ask for reasoning steps
            
        Returns:
            Formatted prompt
            
        Example:
            prompt = PromptTemplate.chain_of_thought(
                context="...",
                question="Why is RAG effective?"
            )
        """
        if include_reasoning:
            prompt = f"""Let's approach this step-by-step:

Context:
{context}

Question: {question}

Please think through this carefully:
1. First, identify the relevant information in the context
2. Then, reason through the question systematically
3. Finally, provide a clear answer

Your response:"""
        else:
            prompt = f"""Think through this step-by-step.

Context:
{context}

Question: {question}

Answer (with reasoning):"""
        
        return prompt
    
    # ========================================
    # ReAct PROMPTS (Reasoning + Acting)
    # ========================================
    
    @staticmethod
    def react(
        context: str,
        question: str
    ) -> str:
        """
        ReAct prompting - reasoning and acting.
        
        Args:
            context: Retrieved context documents
            question: User question
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Use the following format:

Thought: Think about what information you need
Action: Identify relevant information from the context
Observation: What did you find?
Thought: Reason about the observation
Answer: Provide the final answer

Context:
{context}

Question: {question}

Let's begin:
Thought:"""
        
        return prompt
    
    # ========================================
    # SPECIALIZED PROMPTS
    # ========================================
    
    @staticmethod
    def comparison(
        context: str,
        items: List[str],
        criteria: Optional[List[str]] = None
    ) -> str:
        """
        Prompt for comparing multiple items.
        
        Args:
            context: Retrieved context
            items: Items to compare
            criteria: Optional comparison criteria
            
        Returns:
            Formatted prompt
        """
        items_text = ", ".join(items)
        
        if criteria:
            criteria_text = "\n".join([f"- {c}" for c in criteria])
            prompt = f"""Based on the context below, compare the following: {items_text}

Consider these criteria:
{criteria_text}

Context:
{context}

Provide a structured comparison:"""
        else:
            prompt = f"""Based on the context below, compare and contrast: {items_text}

Context:
{context}

Comparison:"""
        
        return prompt
    
    @staticmethod
    def summarization(
        context: str,
        max_sentences: Optional[int] = None,
        focus: Optional[str] = None
    ) -> str:
        """
        Prompt for summarizing content.
        
        Args:
            context: Content to summarize
            max_sentences: Maximum number of sentences
            focus: Optional focus area
            
        Returns:
            Formatted prompt
        """
        constraints = []
        if max_sentences:
            constraints.append(f"in no more than {max_sentences} sentences")
        if focus:
            constraints.append(f"focusing on {focus}")
        
        constraint_text = " ".join(constraints) if constraints else ""
        
        prompt = f"""Summarize the following content{' ' + constraint_text if constraint_text else ''}:

Context:
{context}

Summary:"""
        
        return prompt
    
    @staticmethod
    def explanation(
        context: str,
        concept: str,
        audience: str = "general"
    ) -> str:
        """
        Prompt for explaining concepts.
        
        Args:
            context: Context containing information
            concept: Concept to explain
            audience: Target audience (general, expert, beginner)
            
        Returns:
            Formatted prompt
        """
        audience_instructions = {
            "general": "Explain in clear, accessible language for a general audience.",
            "expert": "Provide a technical, detailed explanation for domain experts.",
            "beginner": "Explain as if to someone with no prior knowledge, using simple terms and analogies."
        }
        
        instruction = audience_instructions.get(audience, audience_instructions["general"])
        
        prompt = f"""{instruction}

Context:
{context}

Explain: {concept}

Explanation:"""
        
        return prompt
    
    # ========================================
    # CITATION & SOURCE TRACKING
    # ========================================
    
    @staticmethod
    def with_citations(
        context: str,
        question: str,
        numbered_sources: bool = True
    ) -> str:
        """
        Prompt that encourages citation of sources.
        
        Args:
            context: Retrieved context (should have numbered documents)
            question: User question
            numbered_sources: Whether sources are numbered
            
        Returns:
            Formatted prompt
        """
        if numbered_sources:
            instruction = "Cite specific document numbers in your answer (e.g., 'According to Document 2...')."
        else:
            instruction = "Reference specific sources in your answer."
        
        prompt = f"""Answer the question based on the context. {instruction}

Context:
{context}

Question: {question}

Answer with citations:"""
        
        return prompt
    
    # ========================================
    # QUALITY CONTROL
    # ========================================
    
    @staticmethod
    def with_confidence(
        context: str,
        question: str
    ) -> str:
        """
        Prompt that asks for confidence assessment.
        
        Args:
            context: Retrieved context
            question: User question
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Answer the question based on the context. At the end, rate your confidence (High/Medium/Low) and explain why.

Context:
{context}

Question: {question}

Answer:
[Your answer here]

Confidence Level: [High/Medium/Low]
Reasoning: [Why this confidence level]"""
        
        return prompt
    
    @staticmethod
    def detect_insufficient_context(
        context: str,
        question: str
    ) -> str:
        """
        Prompt that handles insufficient context gracefully.
        
        Args:
            context: Retrieved context
            question: User question
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Answer the question based ONLY on the information in the context.

If the context doesn't contain enough information to answer fully:
1. Say what you CAN answer based on the context
2. Clearly state what information is missing
3. Suggest what additional context would help

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt


class PromptBuilder:
    """
    Fluent interface for building complex prompts.
    
    Example:
        prompt = (PromptBuilder()
                 .set_context(context)
                 .set_question(question)
                 .set_strategy(PromptStrategy.CHAIN_OF_THOUGHT)
                 .add_instruction("Be concise")
                 .require_citations()
                 .build())
    """
    
    def __init__(self):
        self.context = ""
        self.question = ""
        self.strategy = PromptStrategy.ZERO_SHOT
        self.instructions = []
        self.examples = []
        self.system_prompt = None
        self.require_citations_flag = False
        self.require_confidence = False
    
    def set_context(self, context: str) -> 'PromptBuilder':
        """Set the context."""
        self.context = context
        return self
    
    def set_question(self, question: str) -> 'PromptBuilder':
        """Set the question."""
        self.question = question
        return self
    
    def set_strategy(self, strategy: PromptStrategy) -> 'PromptBuilder':
        """Set the prompting strategy."""
        self.strategy = strategy
        return self
    
    def set_system_prompt(self, prompt_type: str = "default") -> 'PromptBuilder':
        """Set system prompt."""
        self.system_prompt = PromptTemplate.SYSTEM_PROMPTS.get(
            prompt_type,
            PromptTemplate.SYSTEM_PROMPTS["default"]
        )
        return self
    
    def add_instruction(self, instruction: str) -> 'PromptBuilder':
        """Add an instruction."""
        self.instructions.append(instruction)
        return self
    
    def add_examples(self, examples: List[Dict[str, str]]) -> 'PromptBuilder':
        """Add few-shot examples."""
        self.examples = examples
        return self
    
    def require_citations(self) -> 'PromptBuilder':
        """Require citations in the response."""
        self.require_citations_flag = True
        return self
    
    def require_confidence_score(self) -> 'PromptBuilder':
        """Require confidence assessment."""
        self.require_confidence = True
        return self
    
    def build(self) -> tuple[str, Optional[str]]:
        """
        Build the final prompt.
        
        Returns:
            Tuple of (prompt, system_prompt)
        """
        # Combine instructions
        instruction_text = " ".join(self.instructions) if self.instructions else None
        
        # Build based on strategy
        if self.strategy == PromptStrategy.ZERO_SHOT:
            prompt = PromptTemplate.zero_shot(
                self.context,
                self.question,
                instruction_text
            )
        
        elif self.strategy == PromptStrategy.FEW_SHOT:
            prompt = PromptTemplate.few_shot(
                self.context,
                self.question,
                self.examples,
                instruction_text
            )
        
        elif self.strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            prompt = PromptTemplate.chain_of_thought(
                self.context,
                self.question
            )
        
        elif self.strategy == PromptStrategy.REACT:
            prompt = PromptTemplate.react(
                self.context,
                self.question
            )
        
        else:
            prompt = PromptTemplate.zero_shot(
                self.context,
                self.question,
                instruction_text
            )
        
        # Add citations requirement
        if self.require_citations_flag:
            prompt = prompt.replace(
                "Answer:",
                "Answer (with citations to specific documents):"
            )
        
        # Add confidence requirement
        if self.require_confidence:
            prompt += "\n\nInclude confidence level: High/Medium/Low"
        
        return prompt, self.system_prompt