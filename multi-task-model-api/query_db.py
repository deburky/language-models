"""query_db.py."""

import duckdb


def main():
    try:
        # Connect to the database
        con = duckdb.connect("db/embeddings.db")

        # Query the number of rows in the embeddings table
        result_count = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        print(f"Number of rows in embeddings table: {result_count[0]}")

        # Fetch first 50 rows (or all rows)
        result = con.execute("SELECT * FROM embeddings").fetchmany(2)

        # Print the raw result
        print("Raw result from the query:")
        print(result)

        # Check if result is empty
        if not result:
            print("No data found in embeddings table.")
        else:
            print(f"Loaded {len(result)} rows.")

    except Exception as e:
        print(f"Error: {e}")

    # Save list to txt file
    with open("output.txt", "w") as file:
        for row in result:
            file.write(f"{row}\n")

if __name__ == "__main__":
    main()
