from agentmesh.core.help_catalog import list_help_entries


def test_help_catalog_lists_core_commands() -> None:
    commands = [entry.command for entry in list_help_entries()]

    assert "agentmesh run" in commands
    assert "agentmesh configure-models" in commands
    assert "agentmesh list-models" in commands
