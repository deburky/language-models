"""Aurora PostgreSQL backend — IAM auth connection and pgvector search."""

import os

import boto3
import psycopg2


def _get_connection():
    """Establish a connection to the Aurora PostgreSQL database using IAM authentication."""
    endpoint = os.environ["AURORA_ENDPOINT"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    user = os.environ.get("AURORA_DB_USER", "postgres")
    db = os.environ.get("AURORA_DB_NAME", "postgres")
    token = boto3.client("rds", region_name=region).generate_db_auth_token(
        DBHostname=endpoint,
        Port=5432,
        DBUsername=user,
        Region=region,
    )
    return psycopg2.connect(
        host=endpoint,
        port=5432,
        database=db,
        user=user,
        password=token,
        sslmode="require",
        connect_timeout=10,
    )


def _ensure_schema(conn) -> None:
    """Create the pgvector extension and document_vectors table if they do not exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS document_vectors (
                id          SERIAL PRIMARY KEY,
                user_id     TEXT    NOT NULL,
                document_id TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text  TEXT    NOT NULL,
                embedding   vector(384),
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
    conn.commit()


def _vec(embedding: list[float]) -> str:
    """Convert a list of floats to a string representation for PostgreSQL vector type."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


def write_chunks(
    user_id: str, document_id: str, chunks: list[str], embeddings: list[list[float]]
):
    """Write document chunks and their embeddings, then ensure the IVFFlat index exists."""
    conn = _get_connection()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO document_vectors
                        (user_id, document_id, chunk_index, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    (user_id, document_id, i, text, _vec(embedding)),
                )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS embedding_cosine_idx
                ON document_vectors
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
        conn.commit()
    finally:
        conn.close()


def search(user_id: str, query_embedding: list[float], k: int = 5) -> list[dict]:
    """Search for the top-k most similar document chunks for a given user and query embedding."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_text, document_id, chunk_index,
                       1 - (embedding <=> %s::vector) AS score
                FROM document_vectors
                WHERE user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vec(query_embedding), user_id, _vec(query_embedding), k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "chunk_text": row[0],
            "document_id": row[1],
            "chunk_index": row[2],
            "score": round(float(row[3]), 4),
        }
        for row in rows
    ]
