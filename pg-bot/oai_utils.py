import openai
from config import EMBEDDING_MODEL, COMPLETION_MODEL


client = openai.OpenAI()

def vectorize_data_with_openai(data):
    """Takes in a string and returns a vectorized representation of the string."""
    embed_response = client.embeddings.create(input=data, model=EMBEDDING_MODEL)
    return embed_response.data[0].embedding


def get_completion_from_messages(
    messages, model=COMPLETION_MODEL, temperature=0, max_tokens=4000
):
    """Returns chat completion from OpenAI API"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # randomness of the model's output
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def get_system_message_for_vector_retrievals(retrieved_data):
    """Returns a system message with the retrieved data embedded

    Args:
        retrieved_data: (list of dicts) of data retrieved from the labnetwork
        user forum
    Returns:
        system_message: (str) formatted string with the retrieved data included
    """

    system_message = f"""
        You are an engineer with expertise in complex tools and also have
        expert capabilities knowing how to run a nanofabrication facility.

        You will receive a list of data retrieved from the labnetwork user forum, 
        each element which will be formatted as a dictionary containing the ranking of
        the data, the sender of the data, and the body of the text message itself.
        lower ranks (numbers) are more relevant 

        Base your answers to the user prompt only on the retrieved data below:
        
        #### Retrieved Labnetwork Data ####
        {retrieved_data}

        Formulate a response that best matches the user's query, 
        Give the response with as much relevant detail as possible
        Do not preface or end the response with extra polite words. 
        Just answer the question with the facts. Format the response as the 
        user would like to see it if specified.  
        
        Do not answer questions that are not relevant to the data that is retrieved.
        If the retrieved texts do not contain any information to be able
        to answer the user query, you must reply that you do not have the 
        necessary information, and that the user should ask a relevant labnetwork
        question.  
        
        """
    return system_message
