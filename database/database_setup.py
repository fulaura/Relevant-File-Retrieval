from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2 import sql

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def db_connect():
    try:
        conn = psycopg2.connect(
            dbname="RF_Retrieval", 
            user="postgres", 
            password="1234", 
            host="localhost",  
            port="5432"
        )
        cur = conn.cursor()
        cur.execute("SET CLIENT_ENCODING TO 'UTF8';")
        conn.commit()
        print("Database connected successfully!")
        return conn,cur
    except Exception as e:
        conn.rollback()
        print("Unexpected error occured while connecting:", e)
        return None, None

from utils.run_once import run_once
@run_once(flag_file="__run_once_flags__/db_init.flag")
def db_init(conn,cur):
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        conn.commit()

        cur.execute("""
            CREATE TABLE documents (
                id SERIAL PRIMARY KEY,
                
                file_name TEXT NOT NULL,
                file_type TEXT,
                
                created_date TIMESTAMP,
                modified_date TIMESTAMP,
                
                path TEXT NOT NULL,
                inserted_at TIMESTAMP DEFAULT NOW(),

                metadata JSONB DEFAULT '{}'::jsonb,
                content TEXT,
                
                embedding vector(768)
            );
        """)
        conn.commit()
        print("Table 'documents' created successfully!")
    except Exception as e:
        conn.rollback()
        print("Unexpected error occured while Setting Up the Database:", e)
        return None, None

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def document_exists(cur, path: str) -> bool:
    cur.execute("SELECT 1 FROM documents WHERE path = %s LIMIT 1;", (path,))
    return cur.fetchone() is not None


def db_load(conn, cur, data: list[dict]):
    insert_query = """
        INSERT INTO documents (
            file_name,
            file_type,
            created_date,
            modified_date,
            path,
            metadata,
            content,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """

    rows = []
    for doc in data:
        rows.append((
            doc.get("file_name"),
            doc.get("file_type"),
            doc.get("created_date"),
            doc.get("modified_date"),
            doc.get("path"),
            json.dumps(doc.get("metadata", {}), default=convert_numpy),
            doc.get("content"),
            doc.get("embedding")
        ))

    try:
        cur.executemany(insert_query, rows)
        conn.commit()
        print(f"{len(rows)} chunks inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data: {e}")
        
#TODO: when enough data collected
def db_init_index(conn, cur):
    try:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_ivfflat_cosine
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        conn.commit()
        print("IVFFlat index created successfully!")
    except Exception as e:
        conn.rollback()
        print("Error while creating index:", e)


if __name__ == "__main__":
    conn, cur = db_connect()
    if conn and cur:
        db_init(conn, cur)
        # Example data to load
        example_data = [
            {
                "file_name": "example.pdf",
                "file_type": "pdf",
                "created_date": "2023-10-01 12:00:00",
                "modified_date": "2023-10-02 12:00:00",
                "content": "This is an example content.",
                "metadata": {"author": "John Doe"},
                'path': "path/to/example.pdf",
                "embedding": np.random.rand(768).tolist()
            }
        ]
        db_load(conn, cur, example_data)
        cur.close()
        conn.close()