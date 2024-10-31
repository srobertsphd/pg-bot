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

from config import DATABASE_URL
########################################################################

def connect_db():
    """Connect to the PostgreSQL database and register the vector extension."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


########################################################################
# Functions needed for creating and filling the table

def enable_pgvector_extension(cur):
    """Enable the pgvector extension when creating table

    """
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    print("pgvector extension enabled")



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
            query = """
                SELECT chunk_id, parent_type, parent_content, 
                       name_of_tool, content
                FROM chunk_embeds
                ORDER BY embedding <=> %s
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


########################################################################

def list_tables():
    """List all tables in the database."""
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            # Query to get all user tables (excluding system tables)
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [table[0] for table in cur.fetchall()]
            
        conn.close()
        return tables
        
    except Exception as e:
        print(f"Error listing tables: {e}")
        return []
    

def list_columns(table_name):
    """
    List all columns in the specified table.
    
    Args:
        table_name (str): Name of the table to query
        
    Returns:
        list: List of column names in the table
    """
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            columns = [(col[0], col[1]) for col in cur.fetchall()]
            
        conn.close()
        return columns
        
    except Exception as e:
        print(f"Error listing columns for table {table_name}: {e}")
        return []
    

def get_tool_names():
    """
    Get all tool names from the tool_table.
    
    Returns:
        list: List of tool names from the database
    """
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            query = """
                SELECT tool_name 
                FROM tool_table
                ORDER BY tool_name;
            """
            cur.execute(query)
            # Convert list of tuples to simple list
            tool_names = [tool[0] for tool in cur.fetchall()]
            
        conn.close()
        return tool_names
        
    except Exception as e:
        print(f"Error getting tool names: {e}")
        return []