import psycopg
from pgvector.psycopg import register_vector

DB_URL = "postgresql://postgres:postgres@localhost:5432/active_learning"

#connect to the db (register_vector allows numpy insertion and querying)
def get_conn(register_vec: bool = True):
    con = psycopg.connect(DB_URL)
    if register_vec:
        register_vector(con)
    return con

#initialize db tables & sample index
def init_db():
    with get_conn(register_vec=False) as conn:
        with conn.cursor() as cur:
            #make sure vector datatype exists in postgres instance
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            #create table to hold relevant sample information
            cur.execute("CREATE TABLE samples (" \
            "               id SERIAL PRIMARY KEY" \
            "               title TEXT NOT NULL" \
            "               body TEXT NOT NULL" \
            "               embedding VECTOR(384) NOT NULL" \
            "               umap_x REAL NOT NULL" \
            "               umap_y REAL NOT NULL" \
            "               human_label INTEGER DEFAULT NULL)")
            #create iterations table to track learning progress
            cur.execute("CREATE TABLE iterations (" \
            "               id SERIAL PRIMARY KEY" \
            "               iteration_num INTEGER NOT NULL" \
            "               labeled_count INTEGER NOT NULL" \
            "               mean_entropy REAL NOT NULL" \
            "               created_at TIMESTAMPTZ DEFAULT NOW() )")
            #create searchable indexes using HNSW (makes searching way more efficient in higher dimensions)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_embedding "\
            "            ON samples USING hnsw (embedding vector_cosine_ops);")

#resets db tables to default
def reset_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE samples, iterations RESTART IDENTITY;")

