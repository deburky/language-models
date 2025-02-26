import subprocess
import sys
from rich.console import Console

console = Console()

# Define scripts with correct arguments
scripts_with_args = [
    ("run_fine_tuning_experiments.py", ["--batch-size", "32", "--epochs", "2"]),
    ("run_supervised_evaluation.py", ["--pooling", "cls", "--n-estimators", "100"]),
]

for script, args in scripts_with_args:
    console.print(f"\n🚀 [bold cyan]Running {script}...[/bold cyan]\n")

    # Construct the full command
    command = [sys.executable, script] + args

    # Run the script and stream output live
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, text=True)

    # Wait for script to finish
    process.wait()

    # Check if the script failed
    if process.returncode != 0:
        console.print(f"[red]Error while running {script}, stopping pipeline.[/red]")
        sys.exit(1)

console.print("\n✅ [bold green]Pipeline completed successfully![/bold green] 🎉")
