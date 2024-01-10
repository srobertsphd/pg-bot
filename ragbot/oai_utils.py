import openai
from config import OPENAI_EMBEDDING_MODEL, OPENAI_COMPLETION_MODEL

delimiter = "####"

def vectorize_data_with_openai(data):
    """
    This returns the vector embedding only, without the other
    components of the openia object - (this is the ['data'])
    """
    vector = openai.Embedding.create(input=data, engine=OPENAI_EMBEDDING_MODEL)
    return vector['data'][0].embedding


def get_completion_from_messages(messages, 
                                 model=OPENAI_COMPLETION_MODEL,
                                 temperature=0,
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # randomness of the model's output
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]


def get_system_tool_message(retrieved_texts, tool_name_description):
    
    system_message = f"""
        You are an engineer with expertise in complex tools
        Follow these instructions to process the user query. 
        The user query is delimited with {delimiter}.

        [Context from Vector Database]
        These {retrieved_texts} are the top relevant pieces of information 
        retrieved from the vector database. 
        Please use this along with your ability to search outside of the 
        retrieved texts provided. 
        The name of the equipment that the questions are relation to is
        {tool_name_description}

        [Instructions]
            
        {delimiter} Formulate a response that best matches the user's query, 
        Give the response with as much relevant detail as possible
        Do not preface or end the response with extra polite words. 
        Just answer the question with the facts. 
        If the retrieved texts do not contain any information to be able
        to answer the user query, then reply that you do not have the 
        necessary information.  
        
        """
    return system_message