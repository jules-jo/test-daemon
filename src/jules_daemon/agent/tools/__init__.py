"""Agent tool implementations wrapping existing daemon functionality.

Each tool is a thin adapter that delegates to an existing daemon module.
No business logic is reimplemented -- tools compose existing functions
from wiki, execution, monitor, and SSH sub-packages.

Tool catalog (10 baseline tools):

Read-only (ApprovalRequirement.NONE):
    read_wiki           -- wiki.command_translation + wiki.test_knowledge
    lookup_test_spec    -- wiki.test_knowledge (structured test catalog)
    check_remote_processes -- execution.collision_check
    read_output         -- wiki.current_run + wiki.state_reader
    parse_test_output   -- monitor.test_output_parser

State-changing (ApprovalRequirement.CONFIRM_PROMPT):
    propose_ssh_command -- proposes command for human approval
    execute_ssh         -- execution.run_pipeline (requires prior approval)

User interaction:
    ask_user_question   -- IPC question to user for missing arguments
    summarize_run       -- execution.output_summarizer
    notify_user         -- IPC notification push to CLI subscribers
"""

from jules_daemon.agent.tools.ask_user_question import AskUserQuestionTool
from jules_daemon.agent.tools.check_remote_processes import (
    CheckRemoteProcessesTool,
)
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool
from jules_daemon.agent.tools.notify_user import NotifyUserTool
from jules_daemon.agent.tools.parse_test_output import ParseTestOutputTool
from jules_daemon.agent.tools.propose_ssh_command import ProposeSSHCommandTool
from jules_daemon.agent.tools.read_output import ReadOutputTool
from jules_daemon.agent.tools.read_wiki import ReadWikiTool
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool

__all__ = [
    "AskUserQuestionTool",
    "CheckRemoteProcessesTool",
    "ExecuteSSHTool",
    "LookupTestSpecTool",
    "NotifyUserTool",
    "ParseTestOutputTool",
    "ProposeSSHCommandTool",
    "ReadOutputTool",
    "ReadWikiTool",
    "SummarizeRunTool",
]
