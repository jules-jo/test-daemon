"""Agent loop infrastructure for the Jules SSH Test Runner Daemon.

This package implements the iterative think-act cycle that replaces
the one-shot LLM translation path. The LLM receives a ToolRegistry of
tools it can call iteratively, observe results (including failures),
and propose corrections -- all while requiring explicit human approval
for every state-changing SSH operation.

Sub-modules:
    agent_loop           -- AgentLoop state machine: orchestrates the
                            THINKING -> ACTING -> OBSERVING cycle with
                            max-iteration guard and termination conditions.
    conversation_history -- Immutable conversation history accumulator
                            with pure append helpers for system, user,
                            assistant, and tool-result messages.
    llm_adapter          -- OpenAI SDK adapter with the core LLM API
                            call wrapper (call_completion) handling retry
                            logic, timeout enforcement, and error
                            classification for the agent loop's think phase.
    response_parser      -- Discriminated union response parser that
                            classifies raw LLM output as either a
                            ToolCallsResponse (tool invocation requests)
                            or a FinalAnswerResponse (text answer).
                            Supports both native and prompt-based modes.
    tool_types           -- Protocol types: ToolCall, ToolResult, ToolParam,
                            ToolSpec, ApprovalRequirement, ToolResultStatus
    tool_result          -- Convenience re-export for ToolResult and ToolResultStatus.
    tool_registry        -- ToolRegistry: stores Tool instances by name,
                            validates calls against schemas, serializes to
                            OpenAI-compatible function schemas.
    tool_dispatch        -- ToolDispatchBridge: bridges parsed tool-call
                            requests from the response parser to the
                            ToolRegistry, collects execution results
                            (including error/failure observations), and
                            formats them back as observation messages
                            appended to conversation history. Handles
                            the ACTING and OBSERVING phases of the loop.
"""
