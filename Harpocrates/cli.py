import click
from Harpocrates.scanner.detector import detect_entropy_secrets
from Harpocrates.utils.file_loader import scan_file_for_entropy
from Harpocrates.scanner.entropy import shannon_entropy

@click.group()
def cli():
    pass

@cli.command()
@click.argument("text")
def entropy(text):
    found, score = detect_entropy_secrets(text)
    click.echo(f"Entropy: {score:.3f} | Secret Found: {found}")
    
@cli.command()
@click.argument("filepath")
def scan(filepath):
    results = scan_file_for_entropy(filepath)
    if not results:
        click.echo("No potential secrets found.")
        return
    click.echo("Potential secrets found:\n")
    for r in results:
        click.echo(f"Line {r['line']}: {r['token']}")

if __name__ == "__main__":
    cli()