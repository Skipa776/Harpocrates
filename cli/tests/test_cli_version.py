import pytest
from cli_entry import COMMAND_NAMES, CommandNameRunner

from Harpocrates import __version__
from Harpocrates.cli import app

runner = CommandNameRunner()


@pytest.fixture(autouse=True, params=COMMAND_NAMES)
def _command_name(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-NAME-02: every test in this file runs once as `harpo` and once as the deprecated `harpocrates`."""
    runner.use(request.param, monkeypatch)


def test_version_flag_prints_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"harpocrates {__version__}"


def test_version_subcommand_still_works() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()
