from typer.testing import CliRunner

from Harpocrates import __version__
from Harpocrates.cli import app

runner = CliRunner()


def test_version_flag_prints_and_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"harpocrates {__version__}"


def test_version_subcommand_still_works() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()
