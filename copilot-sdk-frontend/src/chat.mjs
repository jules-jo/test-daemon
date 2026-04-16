import { CopilotClient } from "@github/copilot-sdk";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rl = readline.createInterface({ input, output });
const DEFAULT_JULES_MCP_TIMEOUT_MS = 180_000;

function denyNonMcpPermissions(request) {
  if (request.kind === "mcp") {
    return { kind: "approved" };
  }
  return {
    kind: "denied-by-permission-request-hook",
    message: "This frontend is restricted to Jules MCP tools.",
    interrupt: true,
  };
}

async function promptUserInput(request) {
  const suffix = request.choices?.length
    ? ` (${request.choices.join(", ")})`
    : "";
  const answer = (await rl.question(`${request.question}${suffix}\n> `)).trim();
  return {
    answer,
    wasFreeform: true,
  };
}

function buildJulesServerConfig(repoRoot) {
  const timeoutMs = Number.parseInt(
    process.env.JULES_MCP_TIMEOUT_MS || `${DEFAULT_JULES_MCP_TIMEOUT_MS}`,
    10,
  );
  return {
    type: "local",
    command: process.env.JULES_MCP_COMMAND || path.join(repoRoot, ".venv", "bin", "python"),
    args: process.env.JULES_MCP_ARGS
      ? process.env.JULES_MCP_ARGS.split(" ")
      : ["-m", "jules_daemon.mcp_server"],
    env: {
      ...process.env,
      PYTHONPATH: process.env.PYTHONPATH || path.join(repoRoot, "src"),
    },
    cwd: repoRoot,
    tools: ["*"],
    timeout: Number.isFinite(timeoutMs) && timeoutMs > 0
      ? timeoutMs
      : DEFAULT_JULES_MCP_TIMEOUT_MS,
  };
}

function buildAgentPrompt() {
  return [
    "You are the Jules Copilot frontend.",
    "Use Jules MCP tools as the source of truth for test knowledge, current workflow state, and history.",
    "Prefer jules_chat for free-form questions about tests and workflow context.",
    "Use jules_status for explicit current-state checks and jules_history for recent results.",
    "This first integration slice does not yet expose workflow execution tools over MCP.",
    "If the user asks to execute or discover tests, explain that the current Copilot SDK bridge is read/control-first and that backend execution wiring is the next step.",
    "Do not invent test metadata that Jules MCP did not return.",
  ].join(" ");
}

async function main() {
  const repoRoot = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "..",
  );
  const model = process.env.COPILOT_MODEL || "gpt-5";

  const client = new CopilotClient();
  await client.start();

  try {
    const session = await client.createSession({
      model,
      workingDirectory: repoRoot,
      streaming: true,
      enableConfigDiscovery: false,
      onPermissionRequest: denyNonMcpPermissions,
      onUserInputRequest: promptUserInput,
      mcpServers: {
        jules: buildJulesServerConfig(repoRoot),
      },
      customAgents: [
        {
          name: "jules",
          displayName: "Jules",
          description: "Chat-first frontend for Jules via MCP",
          prompt: buildAgentPrompt(),
          mcpServers: {
            jules: buildJulesServerConfig(repoRoot),
          },
        },
      ],
      agent: "jules",
    });

    output.write(`Jules Copilot frontend ready (model=${model}). Type 'exit' to quit.\n`);

    while (true) {
      const prompt = (await rl.question("copilot-jules> ")).trim();
      if (!prompt) {
        continue;
      }
      if (["exit", "quit", "q"].includes(prompt.toLowerCase())) {
        break;
      }

      const reply = await session.sendAndWait({ prompt });
      const content = reply?.data?.content || "(no assistant response)";
      output.write(`\n${content}\n\n`);
    }

    await session.disconnect();
  } finally {
    rl.close();
    await client.stop();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
