import os
from typing import Optional

class Config:
    """Configuration object to store and manage application settings."""
    
    def __init__(self):
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        
    @property
    def is_openai_configured(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None and self.openai_api_key.strip() != ""
    
    def validate(self) -> None:
        """Validate that required configuration is present."""
        if not self.is_openai_configured:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running the application."
            )

# Create a singleton instance
config = Config()

