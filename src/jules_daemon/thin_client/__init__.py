"""Minimal example thin client for the Jules daemon IPC interface.

Demonstrates that any process can connect to the daemon via the same
Unix domain socket protocol used by the primary CLI. Serves as both
a proof of extensibility and an integration test harness.

The thin client reuses the shared framing, socket discovery, and
client connection modules but provides its own simplified command
interface -- no classifier, no NL processing, just direct verb-level
IPC exchanges.

Architecture::

    ThinClient
        |
        +-- ClientConnection (shared with primary CLI)
        |       |
        |       +-- socket_discovery (shared)
        |       +-- framing (shared)
        |
        +-- CommandBuilder (verb-specific envelope factories)
        |
        +-- ResponseRenderer (minimal text output)

Usage::

    from jules_daemon.thin_client.client import ThinClient, ThinClientConfig

    config = ThinClientConfig(socket_path="/run/jules/daemon.sock")
    client = ThinClient(config=config)

    # One-shot command
    result = await client.status()
    print(result)

    # Or use the interactive REPL
    await client.repl()
"""
