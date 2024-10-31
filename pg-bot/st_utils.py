import streamlit as st

def clear_messages():
    """Clears the messages and history"""
    st.session_state.messages = []

def clear_history():
    """Clears the history flag and response history list"""
    if 'history' in st.session_state:
        del st.session_state['history']

def display_retrieved_texts(retrieved_texts):
    """Displays the retrieved tests for pulldown under response text"""
    
    for response in retrieved_texts:
        page_number = response['page_number']
        score = response['score']
        text = response['text']
        st.markdown(f"**Page Number:** {page_number}")
        st.markdown(f"**Score:** {score}")
        st.markdown(f"**Text:** {text}")
        st.markdown("---") 
        
# def format_retrieved_chunks(retrieved_chunks):
#     """
#     Format retrieved chunks for display, showing only the content with double line breaks.
    
#     Args:
#         retrieved_chunks: List of tuples from database query
        
#     Returns:
#         str: Formatted string with content elements separated by double line breaks
#     """
#     # Extract only the last element (content) from each tuple and join with double newlines
#     formatted_text = "\n\n".join(chunk[-1] for chunk in retrieved_chunks)
#     return formatted_text

def format_retrieved_chunks(retrieved_chunks):
    """
    Format retrieved chunks for display, showing only the content with horizontal lines.
    
    Args:
        retrieved_chunks: List of tuples from database query
        
    Returns:
        str: Formatted string with content elements separated by markdown horizontal rules
    """
    # Join chunks with horizontal rule markdown syntax
    formatted_text = "\n\n---\n\n".join(chunk[-1] for chunk in retrieved_chunks)
    return formatted_text

# def format_retrieved_texts(retrieved_texts):
#     retrieved_formatted = "\n\n---\n".join(
#         f"Page Number: {item['page_number']} --- Text: {item['text']}" 
#         for item in retrieved_texts
#     )
#     return retrieved_formatted
