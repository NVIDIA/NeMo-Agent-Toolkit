from enum import Enum


class Provider(Enum):
    OPENAI = "openai"
    NIM = "nim"
    ANTHROPIC = "anthropic"
    UNKNOWN = "unknown"
