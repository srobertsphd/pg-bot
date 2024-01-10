"""
This module manages interactions with a Weaviate client. It includes
creating and deleting classes, managing tenants, adding PDF data objects,
and querying data using vector similarity searches.

Functions:
    create_class(class_name): Creates a new class in Weaviate.
    delete_class(class_name): Deletes a class from Weaviate.
    get_schema(): Retrieves the current schema from Weaviate.
    add_tenant(tenant_name): Adds a new tenant to a predefined class.
    remove_tenant(class_name, tenant_name): Removes a tenant from a class.
    write_tenants(class_name): Retrieves and returns sorted tenant names.
    add_pdf_data_objects(df, classname, tenant_name, filename):
        Adds PDF data to Weaviate in batches from a DataFrame.
    _format_query_result(result, class_name):
        Formats query results from Weaviate.
    query_weaviate(input_prompt, k, class_name, tenant_name):
        Performs a vector similarity search and returns top 'k' results.

More detailed descriptions of function parameters, expected data types,
and return values are provided in individual function docstrings.
"""

from weaviate import Tenant
import oai_utils as oai_utils
from config import get_client

client = get_client()

def create_class(class_name):
    schema = {
        "class": class_name,
        "multiTenancyConfig": {"enabled": True},
        "description": "pdfs of technical manuals",
        "vectorizer": "none",
        "properties": [
            {
                "name": "filename",
                "description": "PDF file name",
                "dataType": ["text"]
            },
            {
                "name": "page_number",
                "dataType": ["int"],
            },
            {
                "name": "content",
                "dataType": ["text"]
            },
            {
                "name": "chunk_type",
                "description": "the chunked data type",
                "dataType": ["text"]
            },
            {
                "name": "chunk_number",
                "dataType": ["int"]
            }
        ]
    }
    client.schema.create_class(schema)

def delete_class(class_name):
    client.schema.delete_class(class_name)

def get_schema(class_name):
    return client.schema.get(class_name)

def add_tenant(tenant_name, class_name):
    client.schema.add_class_tenants(
        class_name=class_name,  
        tenants=[Tenant(name=tenant_name)]
    )

def remove_tenant(class_name, tenant_name):
    client.schema.remove_class_tenants(
        class_name=class_name,
        tenants=[tenant_name]
    )

def write_tenants(class_name):
    tenant_list = []
    for tenant in client.schema.get_class_tenants(class_name):
        tenant_list.append(tenant.name)
    return sorted(tenant_list)

def add_pdf_data_objects(df, classname, tenant_name):
    counter = 0
    client.batch.configure(batch_size=25)

    with client.batch as batch:
        for idx, row in df.iterrows():
            # prints status every 100th vector uploaded
            if (counter %100 == 0):
                print(f"import {counter} / {len(df)}")

            embed = row['embedding']
            batch_data = {
                "content": row['text'],
                "chunk_type": row['chunk_type'],
                "chunk_number": idx,
                "filename": row['filename'],
                "page_number": row['page_number'],
            }

            batch.add_data_object(
                data_object=batch_data, 
                class_name=classname, 
                tenant=tenant_name, 
                vector=embed
            )
            counter = counter + 1

    print("Data Added!")

def _format_query_result(result, class_name):
    class_name = class_name.capitalize()
    # list will hold k entries
    result_list = []
    for item in result['data']['Get'][class_name]:
        item_dict = {
            "score": item['_additional']['certainty'],
            "text": item['content'],
            "page_number": item['page_number'],
            # "filename": item['filename'],
        }
        result_list.append(item_dict)
    return result_list

def query_weaviate(input_prompt, k, class_name, tenant_name):
    # vector semantic similarity search
    prompt_embedding = oai_utils.vectorize_data_with_openai(input_prompt)
    vec = {"vector": prompt_embedding}
    result = (
        client 
        .query.get(
            class_name, 
            ["content", 
             "_additional {certainty}", 
             "page_number", 
             "filename"
            ]
        ) 
        .with_near_vector(vec) 
        .with_limit(k) 
        .with_tenant(tenant_name) 
        .do()
    )
    return _format_query_result(result, class_name)