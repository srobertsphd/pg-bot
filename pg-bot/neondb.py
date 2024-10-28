# File: neondb.py
# Author: Samantha Roberts
# Created on: 6/10/2024
# Last modified by: 
# Last modified on: 
#
# Description:
# ------------------------------------------------------------
# This module provides functionalities for interacting with a PostgreSQL database.
# It includes functions to connect to the database, create and manage tables,
# insert and retrieve data, and perform vector-based searches using the pgvector
# extension. This module is used primarily for operations related to the
# 'labnetwork' table, which stores various pieces of information and embeddings
# for the MIT Labnetwork email forum


import os
import sys
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import numpy as np
import math

# Ensure the src directory is in the Python path to import the config module
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from config import DATABASE_URL, SQL_DB_BRANCH_URL

########################################################################

def connect_db():
    """Connect to the PostgreSQL database and register the vector extension."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def connect_sql_branch_db():
    """Connect to the PostgreSQL database and register the vector extension."""
    conn = psycopg2.connect(SQL_DB_BRANCH_URL)
    register_vector(conn)
    return conn

########################################################################
# Functions needed for creating and filling the table

def enable_pgvector_extension(cur):
    """Enable the pgvector extension when creating table

    """
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    print("pgvector extension enabled")


def create_labnetwork_table():
    """Creates labnetwork table if it does not already exist """
    with psycopg2.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            enable_pgvector_extension(cur)

            create_table_query = """
            CREATE TABLE IF NOT EXISTS labnetwork (
                id SERIAL PRIMARY KEY,
                sender TEXT,
                email TEXT,
                domain TEXT,
                date TIMESTAMP,
                subject TEXT,
                message_id TEXT,
                thread_id TEXT,
                cleaned_body TEXT,
                uuid UUID,
                embed VECTOR(1536)
            );
            """
            cur.execute(create_table_query)
            conn.commit()
    print("Table created successfully.")


def insert_data_to_db(df):
    """Insert data from a DataFrame into the labnetwork table in PostgreSQL."""
    # Define the SQL query for inserting data
    insert_query = """
    INSERT INTO labnetwork (
        sender, email, domain, date, 
        subject, message_id, thread_id, 
        cleaned_body, uuid, embed
    ) VALUES %s
    """
    # Convert the DataFrame to a list of tuples
    data_tuples = [(
        row['sender'], row['email'], row['domain'], row['date'], 
        row['subject'], row['message_id'], row['thread_id'], 
        row['cleaned_body'], row['uuid'], row['embed']
    ) for index, row in df.iterrows()]
    
    # Use a context manager to handle the connection and cursor
    with psycopg2.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            execute_values(cur, insert_query, data_tuples)
            conn.commit()
    print("Data uploaded successfully.")


def check_labnetwork_table_exists():
    """Check if the labnetwork table exists in the database.
    Returns:
        Boolean -- True if exists, False if not
    """
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'labnetwork'
            );
        """)
        exists = cur.fetchone()[0]
    return exists

###############################################################################
# Index the vectors in the labnetwork database
# This may not be the best method, or lead to the fastest searching
# could be updated in the future to be more efficient

def get_number_of_records_in_table(cur):
    """Get the number of records in the labnetwork table"""
    cur.execute("SELECT COUNT(*) as cnt FROM labnetwork;")
    num_records = cur.fetchone()[0]
    return num_records

def calculate_num_lists(num_records):
    """Calculate the number of lists to be used for indexing with ivfflat"""
    num_lists = num_records / 1000
    if num_lists < 10:
        num_lists = 10
    if num_records > 1000000:
        num_lists = math.sqrt(num_records)
    return num_lists

def index_labnetwork_data_with_ivfflat():
    """Index the vectors in the labnetwork table using ivfflat"""
    conn = connect_db()
    cur = conn.cursor()
    num_records = get_number_of_records_in_table(cur)
    print("Number of vector records in table: ", num_records)
    num_lists = calculate_num_lists(num_records)
    print("Number of lists in table: ", num_lists)
    cur.execute(f'CREATE INDEX ON labnetwork USING ivfflat (embed vector_cosine_ops) WITH (lists = {num_lists});')
    print("The vectors in the table are now indexed")
    conn.commit()

###############################################################################
# This is used by the vector search nanobot Labnetwork page

def get_top_k_similar_docs(query_embedding, k):
    """ Query the labnetwork table for the top k similar posts
    
    Args:
        query_embedding (list): embedding of the query vector (currently 1536 d)
        k (int): number of similar posts to return
    Returns:
            top_k_docs (list): list of tuples of the top k similar posts
    """
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            # === Return everything in the table but the vectors ===
            query = """
                SELECT sender, email, domain, date, subject,
                        message_id, thread_id, cleaned_body
                FROM labnetwork
                ORDER BY embed <=> %s
                LIMIT %s
            """
            # === Must convert vector to array for pgvector to work ===
            cur.execute(query, (np.array(query_embedding), k))
            top_k_docs = cur.fetchall()
        return top_k_docs
    except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
        print("Connection lost. Reconnecting...")
        conn = connect_db()
        return get_top_k_similar_docs(query_embedding, k, conn)


##############################################################################
# Functions for Execututing SQL query 
# These are used when calling queeries created by openai LLM
# These are broken down so that a failure can throw an exception and be caught 
# by the retry loop



def ask_database_using_sql(query):
    """Function to query postgreSQL database with a provided SQL query.
    
    Args:
        query (str): a fully formed SQL query.
    Returns:
        list -- list of tuples returned by the query
        (actually seems like a string formatted as a list of tuples)
    """
    conn = connect_sql_branch_db()
    cur = conn.cursor()
    try:
        cur.execute(query)
        results = str(cur.fetchall())
        return results # this is a change
    except Exception as e:
        results = f"query failed with error: {e}"
        raise e # this is a change 
    # return results



def execute_sql_query(tool_query_string):
    try:
        results = ask_database_using_sql(tool_query_string)
        # print(f"executed sql query results: {results}")
        return results
    except Exception as e:
        results = f"query failed with error: {e}"
        print(results)
        raise e  # Let the retry mechanism handle the exception
    

def get_items_from_column(col_name, value):
    query = f"""
            SELECT * FROM labnetwork WHERE {col_name} = %s;
        """
    conn = connect_sql_branch_db()
    cur = conn.cursor()
    cur.execute(query, (value,))
    return cur.fetchall()