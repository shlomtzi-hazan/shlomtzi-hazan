from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class Conversation:
    messages: List[Message]
    
    def add_message(self, role: Role, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in format expected by OpenAI API."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


@dataclass
class TechGuidelineInfo:
    company_name: str
    tech_stack: List[str]
    coding_conventions: Dict[str, Any]
    best_practices: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TechGuidelineInfo':
        return cls(
            company_name=data.get('company_name', ''),
            tech_stack=data.get('tech_stack', []),
            coding_conventions=data.get('coding_conventions', {}),
            best_practices=data.get('best_practices', [])
        )
