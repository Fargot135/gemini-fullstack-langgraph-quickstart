"""
Оновлена конфігурація для використання Groq Llama 3.3 70B
"""
import os
from pydantic import BaseModel, Field
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""
    
    query_generator_model: str = Field(
        default="llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the language model to use for the agent's query generation. "
                          "Options: llama-3.3-70b-versatile, llama-3.1-70b-versatile"
        },
    )
    reflection_model: str = Field(
        default="llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the language model to use for the agent's reflection. "
                          "Options: llama-3.3-70b-versatile, llama-3.1-70b-versatile"
        },
    )
    answer_model: str = Field(
        default="llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the language model to use for the agent's answer. "
                          "Options: llama-3.3-70b-versatile, llama-3.1-70b-versatile"
        },
    )
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )
    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)