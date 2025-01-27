# ********** IMPORT LIBRARY *************
import re
import json

from langchain_core.messages import AIMessage

# ********** Function Helper for clean json output after LLM invoking with JSONOutputParser
def json_clean_output(result: AIMessage) -> dict:
    """
    Cleans and parses the output from an AIMessage object to ensure it is in a structured 
    dictionary format. Handles various cases where the AI's response might not be directly 
    formatted as JSON.

    Args:
        result (AIMessage): The AI's output, either in dictionary format or as a string that 
                            might contain JSON content.

    Returns:
        dict: A cleaned and parsed dictionary representation of the AI's response.

    Raises:
        Exception: If JSON parsing fails, the exception is raised and handled internally.
    """
    # ****** Check if the result is already in dictionary format
    if not isinstance(result, dict):
        # Extract content if the result is not a dictionary
        result = result.content
        
        # ****** Attempt to parse the content as JSON
        try:
            clean_respon = json.loads(result.content)
        except Exception as e:
            # ****** Handle cases where JSON content is embedded in the output
            json_content = re.search(r'({.*})', result, re.DOTALL)
                
            if json_content:
                # Extract the JSON content from the matched group
                json_content = json_content.group(1).strip()
            else:
                # Use raw output if JSON extraction fails
                json_content = result
                
            # ****** Parse the extracted JSON content into a dictionary
            clean_respon = json.loads(json_content)
    else:
        # Use the result directly if it's already a dictionary
        clean_respon = result

    return clean_respon