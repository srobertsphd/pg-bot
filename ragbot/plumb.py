"""
This module provides functions to extract text and tables from PDF files,
split text into chunks, and generate embeddings using OpenAI's model. The
extracted data is organized into Pandas dataframes for further analysis.

Functions:
    split_into_chunks(text, chunk_size): Splits text into chunks.
    extract_text_from_pdf(pdf_path, chunk_size): Extracts text from PDF.
    extract_tables_from_pdf(pdf_file_path): Extracts tables from PDF.
    generate_text_df_with_embeddings(chunks): Embeds text chunks.
    generate_table_df_with_embeddings(table_list): Embeds table data.
    combine_text_and_table(text_df, table_df): Combines dataframes.
    pdf_to_df(pdf_path, chunk_size): Pipeline for text/table extraction.

Modules and Libraries:
    - pdfplumber (pdfp): For extracting text/tables from PDFs.
    - pandas (pd): For dataframe manipulation.
    - oai_utils: Utility for OpenAI embedding functionality.

Note:
    Requires OpenAI's embedding models to be configured in 'config'.
"""

import pdfplumber as pdfp
import pandas as pd
import oai_utils

def split_into_chunks(text, chunk_size):
    """Split a text string into chunks separated by a newline character.
    
    Args:
        text (str): The text to split.
        chunk_size (int): The maximum size of each chunk.
    
    Returns:
        list: A list of text chunks.
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []

    for line in lines:
        if sum(len(s) for s in current_chunk) + len(line) < chunk_size and line:
            current_chunk.append(line)
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [line]
            else:
            # If the current line itself exceeds chunk_size, add it as a chunk
                chunks.append(line)
    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append(' '.join(current_chunk))

    return chunks


def extract_text_from_pdf(pdf_path, chunk_size):
    """Extract text from a PDF file and split it into chunks.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int): The maximum number of characters in a chunk.
    
    Returns:
        list: A list of dictionaries, where each dictionary represents a 
        chunk of text with the associated page number.
    """
    
    all_chunks = []

    with pdfp.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text()
                if text:
                    page_chunks = split_into_chunks(text, chunk_size)
                    for chunk in page_chunks:
                        all_chunks.append({'page_number': page.page_number, 
                                           'text': chunk})
            except IndexError:
                print(f"Error extracting text from page {page.page_number}")

    return all_chunks


def extract_tables_from_pdf(pdf_file_path):
    '''Extract tables from a PDF file and return them in a list.
    Args:
        pdf_file_path (str): The path to the PDF file.
    
    Returns:
        list: A list of lists, where each inner list represents a table 
        found in the PDF.
    '''
    tables_list = []

    try:
        with pdfp.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_dict = {
                                'table': table,
                                'page_number': page.page_number
                            }
                            # Append the table to the result list
                            tables_list.append(table_dict)
                    else:
                        # No tables found on this page
                        continue
                except Exception as e:
                    print(f"Error on page {page.page_number}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error while processing the PDF: {str(e)}")

    return tables_list


def generate_text_df_with_embeddings(chunks):
    '''Generate embeddings for text chunks using OpenAI's model.

    Args:
        chunks (list): A list of dicts containing the text and page 
        number of each chunk
    Returns:
        pd.DataFrame: A dataframe containing the embeddings for each 
        chunk along withe the metadata for that chunk
    '''
    embeddings = []
    for chunk in chunks:
        text = chunk['text']
        vector = oai_utils.vectorize_data_with_openai(text)
        embed_dict = {
            'embedding': vector,
            'text': text, 
            'chunk_type': 'text',
            'page_number': chunk['page_number']
        }
        embeddings.append(embed_dict)
    return pd.DataFrame(embeddings)


def generate_table_df_with_embeddings(table_list):
    """Generate dataframe with embeddings for tables using OpenAI.

    Args:
        table_list (list of list): A list of tables, where each table is
        represented as a list of rows.

    Returns:
        list: A list of embeddings for each row in the tables.
    """
    table_embed_list = []
    
    for idx, table in enumerate(table_list):
        for row in table['table']:
            text = " ".join([str(cell) if cell is not None else "" 
                             for cell in row])
            if row != ['']:
                vector = oai_utils.vectorize_data_with_openai(text)
                embed_dict = {
                    'embedding': vector,
                    'text': text, 
                    'chunk_type': 'table',
                    'page_number': table['page_number']
                }
                table_embed_list.append(embed_dict)
            else:
                pass
    return pd.DataFrame(table_embed_list)

def combine_text_and_table(text_df, table_df):
    """
    Combine the text and table dataframes into one dataframe.
    """
    df = pd.concat([text_df, table_df], ignore_index=True)
    return df.reset_index(drop=True) 

def pdf_to_df(pdf_path, chunk_size):
    '''calls the PDF functions to extract and embed text from a PDF

    This first extracts the text, then the tables, embeds both, then 
    then combines the text and tables into a single dataframe.

    Args: 
        pdf_path (str): path to the PDF file
        chunk_size (int): number of characters to include in each chunk

    Returns:
        pandas.DataFrame: a dataframe with the text and tables embedded 
    '''
    chunks = extract_text_from_pdf(pdf_path, chunk_size)
    tables = extract_tables_from_pdf(pdf_path)
    text_df = generate_text_df_with_embeddings(chunks)
    table_df = generate_table_df_with_embeddings(tables)
    return combine_text_and_table(text_df, table_df)