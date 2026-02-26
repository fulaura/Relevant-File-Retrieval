import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

def similar_n(conn, cur, vector, n=5, similarity_threshold=None):
    try:
        query_embedding = np.array(vector, dtype=np.float32)

        register_vector(conn)

        if similarity_threshold is None:
            query = """
                SELECT 
                    embedding <-> %s::vector AS similarity, 
                    file_name, 
                    content, 
                    metadata, 
                    path
                FROM documents
                ORDER BY similarity
                LIMIT %s;
            """
            cur.execute(query, (query_embedding, n))

        else:
            query = """
                SELECT 
                    embedding <-> %s::vector AS similarity, 
                    file_name, 
                    content, 
                    metadata, 
                    path
                FROM documents
                WHERE embedding <-> %s::vector <= %s
                ORDER BY similarity
                LIMIT %s;
            """
            cur.execute(query, (query_embedding, query_embedding, similarity_threshold, n))

        results = cur.fetchall()
        conn.commit()
        return results

    except Exception as e:
        print("Unexpected error occurred while searching:", e)
        conn.rollback()
        return []
