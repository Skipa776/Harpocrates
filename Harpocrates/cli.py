import click
from Harpocrates.scanner.detector import detect_entropy_secrets

@click.group()
def cli():
    pass

@cli.command()
@click.argument("text")
def entropy(text):
    found, score = detect_entropy_secrets(text)
    click.echo(f"Entropy: {score:.3f} | Secret Found: {found}")
    
if __name__ == "__main__":
    cli()