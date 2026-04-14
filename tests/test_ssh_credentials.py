"""Tests for SSH credential resolution (password auth support).

Covers:
    - SSHCredential is frozen and redacts password in repr
    - load_credentials_file parses valid YAML with per-host entries
    - load_credentials_file returns empty dict for missing file
    - load_credentials_file returns empty dict for invalid YAML
    - load_credentials_file warns on too-open file permissions
    - load_credentials_file skips entries with missing/invalid fields
    - resolve_ssh_credentials checks credentials file first
    - resolve_ssh_credentials falls back to JULES_SSH_PASSWORD env var
    - resolve_ssh_credentials returns None when no credentials found
    - Credential source field never contains the actual password
    - JULES_SSH_CREDENTIALS_FILE env var overrides default path
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from jules_daemon.ssh.credentials import (
    REDACTED,
    SSHCredential,
    build_missing_password_guidance,
    load_credentials_file,
    resolve_ssh_credentials,
)


# ---------------------------------------------------------------------------
# SSHCredential model
# ---------------------------------------------------------------------------


class TestSSHCredential:
    """Verify SSHCredential is frozen and redacts passwords."""

    def test_frozen(self) -> None:
        cred = SSHCredential(
            username="deploy", password="secret", source="test"
        )
        with pytest.raises(AttributeError):
            cred.password = "other"  # type: ignore[misc]

    def test_repr_redacts_password(self) -> None:
        cred = SSHCredential(
            username="deploy",
            password="super-secret-password",
            source="test",
        )
        text = repr(cred)
        assert "super-secret-password" not in text
        assert REDACTED in text
        assert "deploy" in text

    def test_str_redacts_password(self) -> None:
        """str() uses repr for dataclasses, so it should also redact."""
        cred = SSHCredential(
            username="deploy",
            password="super-secret-password",
            source="test",
        )
        text = str(cred)
        assert "super-secret-password" not in text

    def test_fields_accessible(self) -> None:
        cred = SSHCredential(
            username="admin", password="pw123", source="env:VAR"
        )
        assert cred.username == "admin"
        assert cred.password == "pw123"
        assert cred.source == "env:VAR"

    def test_username_can_be_none(self) -> None:
        cred = SSHCredential(
            username=None, password="pw", source="test"
        )
        assert cred.username is None


# ---------------------------------------------------------------------------
# REDACTED constant
# ---------------------------------------------------------------------------


class TestRedacted:
    """Verify the REDACTED sentinel is a non-empty string."""

    def test_is_string(self) -> None:
        assert isinstance(REDACTED, str)

    def test_not_empty(self) -> None:
        assert len(REDACTED) > 0


# ---------------------------------------------------------------------------
# load_credentials_file
# ---------------------------------------------------------------------------


class TestLoadCredentialsFile:
    """Test loading per-host credentials from YAML files."""

    def test_valid_file(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
            "    password: secret123\n"
            "  prod:\n"
            "    username: admin\n"
            "    password: prod-pw\n"
        )
        result = load_credentials_file(creds_file)

        assert len(result) == 2
        assert "staging" in result
        assert result["staging"].username == "deploy"
        assert result["staging"].password == "secret123"
        assert "prod" in result
        assert result["prod"].username == "admin"
        assert result["prod"].password == "prod-pw"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_credentials_file(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "empty.yaml"
        creds_file.write_text("")
        result = load_credentials_file(creds_file)
        assert result == {}

    def test_invalid_yaml_returns_empty(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "bad.yaml"
        creds_file.write_text("{{not valid yaml::")
        result = load_credentials_file(creds_file)
        assert result == {}

    def test_no_hosts_key_returns_empty(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "no_hosts.yaml"
        creds_file.write_text("something_else: true\n")
        result = load_credentials_file(creds_file)
        assert result == {}

    def test_hosts_not_a_dict_returns_empty(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "bad_hosts.yaml"
        creds_file.write_text("hosts:\n  - staging\n  - prod\n")
        result = load_credentials_file(creds_file)
        assert result == {}

    def test_skips_entry_without_password(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "no_pw.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
        )
        result = load_credentials_file(creds_file)
        assert result == {}

    def test_skips_entry_with_empty_password(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "empty_pw.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
            '    password: ""\n'
        )
        result = load_credentials_file(creds_file)
        assert result == {}

    def test_skips_non_dict_entry(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "bad_entry.yaml"
        creds_file.write_text(
            "hosts:\n"
            '  staging: "just-a-string"\n'
        )
        result = load_credentials_file(creds_file)
        assert result == {}

    def test_username_optional(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "no_user.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    password: secret\n"
        )
        result = load_credentials_file(creds_file)
        assert len(result) == 1
        assert result["staging"].username is None
        assert result["staging"].password == "secret"

    def test_source_contains_file_path(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
            "    password: secret\n"
        )
        result = load_credentials_file(creds_file)
        assert str(creds_file) in result["staging"].source

    def test_source_never_contains_password(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
            "    password: top-secret-value\n"
        )
        result = load_credentials_file(creds_file)
        assert "top-secret-value" not in result["staging"].source

    def test_mixed_valid_and_invalid_entries(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "mixed.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  good-host:\n"
            "    username: deploy\n"
            "    password: secret\n"
            "  bad-host:\n"
            "    username: deploy\n"
            '  also-bad: "string"\n'
        )
        result = load_credentials_file(creds_file)
        assert len(result) == 1
        assert "good-host" in result

    def test_strips_hostname_whitespace(self, tmp_path: Path) -> None:
        """Hostnames with whitespace should be stripped."""
        creds_file = tmp_path / "ws.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  '  staging  ':\n"
            "    username: deploy\n"
            "    password: secret\n"
        )
        result = load_credentials_file(creds_file)
        assert "staging" in result


# ---------------------------------------------------------------------------
# File permission warning
# ---------------------------------------------------------------------------


class TestFilePermissions:
    """Verify warning on too-open credentials file permissions."""

    def test_warns_on_group_readable(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    password: secret\n"
        )
        creds_file.chmod(0o644)

        with caplog.at_level("WARNING"):
            load_credentials_file(creds_file)

        assert any("too open" in rec.message for rec in caplog.records)

    def test_no_warning_on_600(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    password: secret\n"
        )
        creds_file.chmod(0o600)

        with caplog.at_level("WARNING"):
            load_credentials_file(creds_file)

        permission_warnings = [
            r for r in caplog.records if "too open" in r.message
        ]
        assert len(permission_warnings) == 0


# ---------------------------------------------------------------------------
# Environment variable override for credentials file path
# ---------------------------------------------------------------------------


class TestCredentialsFileEnvVar:
    """JULES_SSH_CREDENTIALS_FILE env var overrides default path."""

    def test_env_var_path_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom_file = tmp_path / "custom_creds.yaml"
        custom_file.write_text(
            "hosts:\n"
            "  custom-host:\n"
            "    username: custom\n"
            "    password: custom-pw\n"
        )
        monkeypatch.setenv("JULES_SSH_CREDENTIALS_FILE", str(custom_file))

        # Call without explicit path -- should use env var
        result = load_credentials_file()
        assert "custom-host" in result
        assert result["custom-host"].password == "custom-pw"

    def test_explicit_path_overrides_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / "env_creds.yaml"
        env_file.write_text(
            "hosts:\n"
            "  env-host:\n"
            "    password: env-pw\n"
        )
        explicit_file = tmp_path / "explicit_creds.yaml"
        explicit_file.write_text(
            "hosts:\n"
            "  explicit-host:\n"
            "    password: explicit-pw\n"
        )
        monkeypatch.setenv("JULES_SSH_CREDENTIALS_FILE", str(env_file))

        result = load_credentials_file(explicit_file)
        assert "explicit-host" in result
        assert "env-host" not in result


# ---------------------------------------------------------------------------
# resolve_ssh_credentials
# ---------------------------------------------------------------------------


class TestResolveSSHCredentials:
    """Test the full credential resolution chain."""

    def test_credentials_file_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Credentials file takes priority over env var."""
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: file-user\n"
            "    password: file-pw\n"
        )
        monkeypatch.setenv("JULES_SSH_PASSWORD", "env-pw")

        result = resolve_ssh_credentials(
            "staging", credentials_file_path=creds_file
        )

        assert result is not None
        assert result.password == "file-pw"
        assert result.username == "file-user"
        assert "credentials_file" in result.source

    def test_env_var_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no credentials file match, fall back to env var."""
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  other-host:\n"
            "    password: other-pw\n"
        )
        monkeypatch.setenv("JULES_SSH_PASSWORD", "env-pw")

        result = resolve_ssh_credentials(
            "staging", credentials_file_path=creds_file
        )

        assert result is not None
        assert result.password == "env-pw"
        assert result.username is None
        assert "JULES_SSH_PASSWORD" in result.source

    def test_env_var_when_no_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When credentials file does not exist, use env var."""
        monkeypatch.setenv("JULES_SSH_PASSWORD", "env-pw")

        result = resolve_ssh_credentials(
            "staging",
            credentials_file_path=tmp_path / "nonexistent.yaml",
        )

        assert result is not None
        assert result.password == "env-pw"

    def test_returns_none_when_no_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns None when no credentials are available at all."""
        monkeypatch.delenv("JULES_SSH_PASSWORD", raising=False)
        monkeypatch.delenv("JULES_SSH_CREDENTIALS_FILE", raising=False)

        result = resolve_ssh_credentials(
            "staging",
            credentials_file_path=tmp_path / "nonexistent.yaml",
        )

        assert result is None

    def test_env_var_not_used_when_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty JULES_SSH_PASSWORD is treated as absent."""
        monkeypatch.setenv("JULES_SSH_PASSWORD", "")

        result = resolve_ssh_credentials(
            "staging",
            credentials_file_path=tmp_path / "nonexistent.yaml",
        )

        assert result is None

    def test_password_not_in_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The source field must never contain the actual password."""
        monkeypatch.setenv("JULES_SSH_PASSWORD", "my-secret-pw")

        result = resolve_ssh_credentials(
            "staging",
            credentials_file_path=tmp_path / "nonexistent.yaml",
        )

        assert result is not None
        assert "my-secret-pw" not in result.source

    def test_credentials_file_password_not_in_source(
        self, tmp_path: Path
    ) -> None:
        creds_file = tmp_path / "creds.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    password: ultra-secret\n"
        )
        result = resolve_ssh_credentials(
            "staging", credentials_file_path=creds_file
        )

        assert result is not None
        assert "ultra-secret" not in result.source


class TestMissingPasswordGuidance:
    def test_guidance_mentions_default_file_and_env_var(self) -> None:
        message = build_missing_password_guidance()
        assert ".jules" in message
        assert "ssh_credentials.yaml" in message
        assert "JULES_SSH_PASSWORD" in message


# ---------------------------------------------------------------------------
# Integration: YAML number and boolean edge cases
# ---------------------------------------------------------------------------


class TestYAMLEdgeCases:
    """YAML may parse some values as non-strings; ensure graceful handling."""

    def test_numeric_password_parsed_as_string(
        self, tmp_path: Path
    ) -> None:
        """YAML may parse bare numbers; password must be a string."""
        creds_file = tmp_path / "numeric.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
            "    password: 12345\n"
        )
        # YAML parses bare 12345 as int, not str
        result = load_credentials_file(creds_file)
        # Should be skipped because password is not a string
        assert "staging" not in result

    def test_boolean_password_skipped(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "bool.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
            "    password: true\n"
        )
        result = load_credentials_file(creds_file)
        assert "staging" not in result

    def test_quoted_numeric_password_accepted(
        self, tmp_path: Path
    ) -> None:
        creds_file = tmp_path / "quoted.yaml"
        creds_file.write_text(
            "hosts:\n"
            "  staging:\n"
            "    username: deploy\n"
            '    password: "12345"\n'
        )
        result = load_credentials_file(creds_file)
        assert "staging" in result
        assert result["staging"].password == "12345"
