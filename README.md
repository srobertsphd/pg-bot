# RAG-bot

A Retrieval Augmented Generation chatbot app that you can customize with your own data. The app uses the following API's/modules/services:
* OpenAI (embedding and generation)
* Weaviate vector database
* Streamlit (App interface)
* Render (App hosting)
* PDF Plumber (text extraction)

## Table of Contents

- [RAG-bot](#rag-bot)
  - [Table of Contents](#table-of-contents)
  - [See the Rag-bot in action](#see-the-rag-bot-in-action)
  - [Downloading the code](#downloading-the-code)
  - [Create a virtual environment](#create-a-virtual-environment)
  - [OpenAI Key](#openai-key)
  - [Initilalizing a Weaviate Cloud database](#initilalizing-a-weaviate-cloud-database)
  - [Customize the config.py file](#customize-the-configpy-file)
  - [Get sample data from .parquet file](#get-sample-data-from-parquet-file)
  - [Creating a Weaviate Client and Class](#creating-a-weaviate-client-and-class)
  - [Create Tenants](#create-tenants)
  - [Upload data to Weaviate](#upload-data-to-weaviate)
  - [Query Weaviate](#query-weaviate)
  - [Now you have done the following](#now-you-have-done-the-following)
  - [Running Streamlit](#running-streamlit)
  - [How to use your own data](#how-to-use-your-own-data)
  - [Acknowledgements](#acknowledgements)



## See the Rag-bot in action

![Streamlit_dashboard_demo](https://github.com/srobertsphd/RAG-bot/assets/69703058/728b16dc-c8ed-42df-8605-e1df0aaa24df)


## Downloading the code

To obtain the source code for RAG-bot, clone the RAG-bot repository:

```bash
git clone https://github.com/srobertsphd/RAG-bot.git 
```
You should now have a directory structure that looks as follows:

![tree_structure](https://i.imgur.com/mDVtsNS.png)

## Create a virtual environment

Make certain you are have Python version 3.9 or higher installed.  Create a virtual environment using:

```bash
python3 -m venv your_env_name
```
Activate the virtual environment using:

```bash
source your_env_name/bin/activate
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## OpenAI Key
The RAG-bot code uses OpenAi for both two steps:
* Text Embedding 
* Text generation 

You will need an OpenAI API key to run the app.  You can get one from [here](https://beta.openai.com/account/api-keys)

This needs to be set this as an environment variable that can be reached with the following command:
```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
You can set this up by adding the following in your `~/.bashrc` file

```bash
export OPENAI_API_KEY = 'your_openai_api_key_here'
```

## Initilalizing a Weaviate Cloud database

You can used Weaviate Cloud Services (WCS) for free for 14 days by setting up a "sandbox".  To set up a sandbox you can follow the instructions [at this google doc here](https://docs.google.com/document/d/1dBIxpzXRiwKs4IXHVrBCd_iAFt6VF8qWQg9gTbM71aU/edit?usp=sharing).  

1. Set up username and password and login to the dashboard
2. Click "Create Cluster"
3. Make certain you have selected the "Sandbox" tab
4. Enter a name for your database
3. Verify that "Enable Authentication?" is set to "YES"
4. Click "Create" in the lower right corner

It may take a few minutes for the database to be created.  Once it is created you will see a "Details" button.  Click on this and copy the folloing information for your code:

* Cluster URL
* API KEY  

![weaviate_setup_gif](https://i.imgur.com/GArviG8.gif)

## Customize the config.py file

The config.py file contains some of the variables that are used in the app.  You will need to set the following variables:

```python
# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_SANDBOX_API_KEY')

# Weaviate Configuration
WEAVIATE_URL = "your Weaviate URL to sandbox here"
```
## Get sample data from .parquet file

Once you have the config files set up with the API keys and your Weaviate sandbox URL, you can then upload the test data to get started.  

The `RAG-bot/data` directory contains a .parquet file with the data from several PDF files that were read, chunked and for which embeddings were already created for.  You can use these so you can trial the code without generating your own text embeddings, to start.  

To read in this data you can use the following commands, which you can also find in the file `ragbot.ipynb` Jupyter notebook

```python
df = pd.read_parquet('RAG-bot/data/text_and_embeds.parquet', engine='pyarrow')
```

If you further explore this dataframe you can get the info:
```python
df.info()
```
which should look like:

![](https://i.imgur.com/ScTyYjm.png)

This dataframe contains the following information
1. `embedding`: the vector with 1536 dimensions created by the 'ada-002' embedding model
2. `text`: the actual text that was given to the embedding model
3. `chunk_type`: notes whether the text was from a table or the main text of he PDF
4. `page_number`: the page numbe of the PDF the content came from
5. `filename`: the name of the tool the manual refers to

You can now create a list of the 9 technical manuals that the data represents using the folling command:

```python
tenant_list = list(df.filename.unique())
tenant_list
```
The output of `tenant_list` should be: 
```python
['ContourGT-I_Profilometer',
 'Nanoscribe_GT',
 'West_Bond_Wedge_Bonder',
 'Samco_Plasma_Cleaner',
 'Woollam_Ellipsometer',
 'AccuThermo_RTA',
 'NovaNano_SEM',
 'Oxford_100_PECVD',
 'Fiji_G2_ALD']
```

We will use this list to create the separate `tenants` in the Weaviate database, with each tenant containing the vectors relevant to the text in each manual.  

## Creating a Weaviate Client and Class

We will use some of the functions from the RAG-bot module

The first two code cells of the jupyter notebook also have the necessary import statements from the `ragbot` module which are as follows:  

```python
import sys
sys.path.append('/home/sng/RAG-bot/ragbot') 

from ragbot.config import get_client 
from ragbot import weav
from ragbot import plumb
from ragbot import utils
import pandas as pd
```

You can then create an instance of the Weaviate Client and create a class name.  For the purpose of this demonstration we will name our weaviate class `Manuals` as the data represents the text from 9 PDFs of technical manuals.

```python
client = get_client()
WEAVIATE_CLASS = 'Manuals'
client.is_ready()
```

The last line will output `True` if the client and class was properly created

You will now create the class using the schema contained in the file `weav.py`. 

```python
weav.create_class(WEAVIATE_CLASS)
```

```python
weav.get_schema(WEAVIATE_CLASS)
```
this will output a json format with the schema

## Create Tenants

using the `tenant_list` created wen you read in the data, we will create the tenants in the `Manuals` database

```python
for tenant in tenant_list:
    weav.add_tenant(tenant, WEAVIATE_CLASS)
```

To verify if these tenants were added you can write:

```python 
weav.write_tenants(WEAVIATE_CLASS)
```
## Upload data to Weaviate

To upload data to Weaviate:
```python
for tenant in tenant_list:
    temp = df[df.filename == tenant]
    print(f'Now adding {tenant}')
    weav.add_pdf_data_objects(temp, 'New_manuals', tenant)
```

## Query Weaviate

The Query to weaviate will do te following things:
1. Get user prompt
2. Send prompt to openai embeddings api and extract the vector
3. Send the vector to Weaviate to find nearest neighbors
4. Get the retrieved text with the associated metadata, scores, page_number and filename

We need to initialize the following variables:

```python
prompt = 'type your questions here'
k = 10 # number of retrievals
tenamt_name = 'your_tenant_name'
weaviate_class = 'your class_name'
```

And then make the following call to verify the embedding and query process is working ok:

```python
retrieved_texts = weav.query_weaviate(prompt, k, weaviate_class, tenant_name) 
```
The retrieved texts should relate to the prompt, and should contain the metadata, including the page number and the score for the relevance of the retrieved information.  

## Now you have done the following

* Set up you OpenAI API Key
* Set up your Weaviate API Key
* Added the URL to your Weaviate Cloud Sandbox to config
* Created a class in Weaviate called 'Manuals' with a schema
* Defined 9 tenants in the Weaviate class
* Extracted data from the .parquet file in the `data` folded
* Uploaded the embeddings along with the text and other metadata to the database
* Tested that you could query the database and get a response, which verifies your ability to access both OpenAI and Weaviate

You are now prepared to run the Streamlit App!

## Running Streamlit

You can run a local instance of the app in your browser from the RAG-bot directory using:
```bash
streamlit run ragbot/dashboard.py
```

A link should become available, or you can follow the directions to copy/past the local host URL into your browser.  You should now see an instance of the Streamlit app in your browser, that looks like the illustration shown in the introduction.  

---

## How to use your own data

To use your own data you would use the following call, including the path to your own PDF file, along with the chunk size.  

```python
your_data_df = plumb.pdf_to_df(pdf_path, chunk_size)
```

This call does the following:
1. Extracts the text from the PDF (using PDF Plumber)
2. Extracts the tables (if any) from the PDF (using PDF Plumber)
3. Generates the embeddings for the text (using OpenAI)
4. Returns this data in a dataframe ready for uploading to Weaviate.  

All you would have to do after this is to creat the tenant in the database and use this tenant name in the upload.  

## Acknowledgements

* This project was built in collaboration with Aakash N S, CEO of the teaching platform, Jovian.com.  Without their support this project would not have been possible.




