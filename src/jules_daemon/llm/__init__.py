"""Dataiku Mesh LLM client wrapper.

Configures the OpenAI Python SDK with custom base_url, authentication,
and model routing for Dataiku Mesh endpoints. Supports both native
OpenAI-style tool calling and prompt-based simulation fallback.
"""

from jules_daemon.llm.client import create_client, create_completion
from jules_daemon.llm.command_context import (
    CommandContext,
    RiskLevel,
    parse_context_response,
)
from jules_daemon.llm.config import LLMConfig, load_config, load_config_from_env
from jules_daemon.llm.context_classifier import (
    ContextClassifier,
    build_context_system_prompt,
    build_context_user_prompt,
    classify_command as classify_command_context,
)
from jules_daemon.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMParseError,
    LLMResponseError,
    LLMToolCallingUnsupportedError,
)
from jules_daemon.llm.models import ModelID, ToolCallingMode, parse_model_id
from jules_daemon.llm.prompts import (
    HostContext,
    PromptConfig,
    build_messages,
    build_system_prompt,
    build_user_prompt,
)
from jules_daemon.llm.intent_classifier import (
    ClassifiedIntent,
    IntentClassifier,
    IntentConfidence,
    build_intent_system_prompt,
    build_intent_user_prompt,
    classify_intent,
    parse_intent_response,
)
from jules_daemon.llm.response_parser import (
    Confidence,
    LLMCommandResponse,
    LLMCommandStep,
    TranslateResult,
    extract_json_from_text,
    parse_llm_response,
    response_to_ssh_commands,
    translate_command,
)

__all__ = [
    "create_client",
    "create_completion",
    "CommandContext",
    "RiskLevel",
    "parse_context_response",
    "ContextClassifier",
    "build_context_system_prompt",
    "build_context_user_prompt",
    "classify_command_context",
    "LLMConfig",
    "load_config",
    "load_config_from_env",
    "LLMError",
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMParseError",
    "LLMResponseError",
    "LLMToolCallingUnsupportedError",
    "ModelID",
    "ToolCallingMode",
    "parse_model_id",
    "HostContext",
    "PromptConfig",
    "build_messages",
    "build_system_prompt",
    "build_user_prompt",
    "Confidence",
    "LLMCommandResponse",
    "LLMCommandStep",
    "TranslateResult",
    "extract_json_from_text",
    "parse_llm_response",
    "response_to_ssh_commands",
    "translate_command",
    "ClassifiedIntent",
    "IntentClassifier",
    "IntentConfidence",
    "build_intent_system_prompt",
    "build_intent_user_prompt",
    "classify_intent",
    "parse_intent_response",
]
