import sqlite3
import sqlite_vss

print(sqlite_vss.vss_loadable_path())
# '/.../venv/lib/python3.9/site-packages/sqlite_vss/vss0'

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
sqlite_vss.load(conn)

print(conn.execute("select vss_version()").fetchone()[0])
# 'v0.1.0'
