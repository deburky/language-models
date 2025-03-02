"""query_db.py."""

import sqlite3
from rich.console import Console
from rich.table import Table

# Connect to SQLite database
conn = sqlite3.connect("db/vss.db")
curs = conn.cursor()

# Fetch multiple rows (change limit as needed)
curs.execute("SELECT * FROM document LIMIT 2")
rows = curs.fetchmany(2)

# print to text file
with open("output.txt", "w") as f:
    f.write(str(rows))

# Initialize Rich console
console = Console()

# Create Rich Table
table = Table(title="📄 Document Embeddings", show_lines=False, style="bold cyan")

# Add rows
for row in rows:
    table.add_row(*map(str, row))  # Convert everything to string

# Print the table
console.print(table)
