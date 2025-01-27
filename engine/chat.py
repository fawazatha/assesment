# ********** IMPORT FRAMEWORK **********
from langchain_core.prompts             import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables           import RunnableParallel 
from pydantic                           import Field, BaseModel
from langchain_core.output_parsers      import JsonOutputParser, StrOutputParser
from langchain_community.callbacks      import get_openai_callback
from langchain_core.messages            import HumanMessage, AIMessage, BaseMessage

# ********** IMPORT LIBRARIES **********
import requests
import sys 
import os
from typing     import List, Dict, Tuple
import logging
from operator   import itemgetter

# ********** IMPORT **********
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup      import SetupApi, LOGGER, LIST_COLUMNS_FILTER
from model.llms import LLM

# ********** IMPORT HELPER **********
from helper.response_error_helper   import json_clean_output

# ********** IMPORT VALIDATOR **********
from validator.data_type_validation import (validate_string_input, 
                                            validate_dict_input, 
                                            validate_int_input,
                                            validate_list_input)



# *************** Expected format for function convert text to filter
class FilterExpect(BaseModel): 
    """
    FilterExpect defines the expected structure for filter_created.
    """
    filter_created: list = Field(description=[{ 
                                               "field_name": "one of the 'filter_used'",
                                               "value_target": "the value of field expect"
                                               }])
    
# *************** Expected format for function generate decision intent
class IntentDetected(BaseModel):
    """
    IntentDetected defines the expected structure for intent_detected.
    """
    intent_detected: list = Field(description=[{"intent": "The detected intent of the user input.",
                                               "reason": "The reason for the detected intent."
                                              }])


# *************** Function helper for help engine to convert chat history to chat messages
def convert_chat_history(chat_history: list) -> list[BaseMessage]:
    """
    Convert chat history to the chat messages for inputted to LLM.

    Args:
        chat_history (list): List of chat messages, each containing human and AI content.

    Returns:
        list: Converted chat history with alternating HumanMessage and AIMessage objects.
    """
    
    # *************** Validate inputs chat_history is alist
    if not validate_list_input(chat_history, 'chat_history', False):
        LOGGER.error("'chat_history' must be a list of message.")
    
    # *************** Initialize formatted history
    history_inputted = []
    
    # *************** Add messages to formatted history
    for chat in chat_history:
        if chat['type'] == 'human':
            history_inputted.append(HumanMessage(content=chat['content']))
        elif chat['type'] == 'ai':
            history_inputted.append(AIMessage(content=chat['content']))
    
    if history_inputted:
        LOGGER.info(f"Chat History is Converted to BaseMessages: {len(history_inputted)} messages")  
    else:
        LOGGER.warning("No Chat History Inputted")

    # *************** Return formatted history
    return history_inputted

def save_chat_history(chat_history: list, user_input: str, response: str) -> list[dict]:
    """
    Save user input and AI response to the chat history.

    Args:
        chat_history (list): The existing chat history.
        user_input (str): The user's input.
        response (str): The AI's response.

    Returns:
        list: Updated chat history.
    """
    chat_history.extend([
    {"type": "human", "content": user_input},
    {"type": "ai", "content": response}
    ])
    return chat_history

def process_history_entry(entry: Dict | BaseMessage) -> Tuple[str, str]:
    """
    Process a single history entry.
    
    Args:
        entry (Dict | BaseMessage): The history entry to process.

    Returns:
        Tuple[str, str]: The human and AI content from the history entry.
    """
    if isinstance(entry, dict):
        return entry['human'], entry['ai']
    return entry.content, entry.content
    
def context_window(chat_history: list[BaseMessage], window_size: int = 2) -> list[BaseMessage]:
    """
    Create a context window of the chat history.

    Args:
        chat_history (list): List of chat messages.
        window_size (int): The size of the context window.

    Returns:
        list: The context window of chat messages.
    """
    # *************** Validate inputs chat_history is alist
    if not validate_list_input(chat_history, 'chat_history', False):
        LOGGER.error("'chat_history' must be a list of message.")
    
    # *************** Validate inputs window_size is an integer
    if not validate_int_input(window_size, 'window_size'):
        LOGGER.error("'window_size' must be an integer.")
    
    # *************** Initialize context window
    context_window = []
    
    # *************** Add messages to context window
    for i in range(0, len(chat_history)):
        if i + 1 >= len(chat_history):
            break 
        
        # *************** Process history entry
        human_msg, ai_msg = process_history_entry(chat_history[i])
        context_window.extend([
            HumanMessage(content=human_msg),
            AIMessage(content=ai_msg)
        ])
    
    # *************** Return context window
    window_start = max(0, len(context_window) - (window_size * 2))
    return chat_history[window_start:]
        

# *************** Template prompt for function convert text to filter
def prompt_convert_text_to_filter() -> PromptTemplate:
    """
    Prompt to generate filter to use with weather api call

    Returns:
        PromptTemplate: _description_
    """
        
    template = """
    You are Filter Creation. Your task is to convert user input into filters. 
    
    Input: 
    "text_input": {text_input}
    "filter_used": {field_names} 
    "chat_history": {chat_history}
    
    Instructions:
    1. Analyze "text_input" for new filters.
    2. Validate all filters against "filter_used".
    3. Use "chat_history" to see if the user has already provided enough information in previous messages.
    
    Output:
    Return filters as a JSON list with "field_name", and "value_target".
    {format_instructions}
    """
    
    return PromptTemplate(
        template=template,
        input_variables=["text_input", "field_names", "chat_history"],
        partial_variables={"format_instructions": JsonOutputParser(pydantic_object=FilterExpect).get_format_instructions()}
    )

# *************** Function to convert text to filter for weather api
def convert_text_to_filter(text_input: str, field_names: list|dict, chat_history:list[dict]) -> dict[list]:
    """
    Convert the user input text into filters to use with the OpenWeather API.

    Args:
        text_input (str): User's input text
        field_names (list | dict): List columns filter to use with the OpenWeather API.

    Returns:
        dict[list]: 
    """
    # *************** Validate inputs text_input not an empty string
    if not validate_string_input(text_input, 'text_input_filter'):
        LOGGER.error("'text_input' must be a string.")
        
    prompt = prompt_convert_text_to_filter()
    
    # ********* Setup Runnable Chain
    runnale_filter = RunnableParallel(
                                {
                                    "text_input": itemgetter('text_input'),
                                    "field_names": itemgetter('field_names'),
                                    "chat_history": itemgetter('chat_history')
                                }
                            )
    
    # ********* chaining prompt with llm
    chain = ( 
             runnale_filter | 
             prompt | 
             LLM()
             )
    
    # ********* Invoke the chain
    filter_response = chain.invoke(
                                    {
                                        "text_input": text_input, 
                                        "field_names": field_names,
                                        "chat_history": chat_history
                                    }
                                )
    
    # ********* Clean the output and logger info
    clean_result = json_clean_output(filter_response)
    LOGGER.info(f"Filter Creation:\n {clean_result}")
    
    return clean_result

# *************** Function to call weather api
def call_weather_api(filters: dict, intent_detected: str) -> List[Dict]:
    """
    Call the OpenWeather API to get the current weather data for multiple locations.

    Args:
        filters (dict): The filters to use in the API call.

    Returns:
        list: A list of responses from the OpenWeather API (one for each input).
    """
    # *************** Validate the filters
    if not validate_dict_input(filters, 'filters'):
        LOGGER.error("'filters' must be a dict.")
        return [{"error": "'filters' must be a non-empty dictionary."}]
    
    required_fields = {"q", "zip", "lat", "lon", "cnt"}
    extracted_params = []
    
    # *************** Flatten filter_created into individual query parameters
    for filter_item in filters.get("filter_created", []):
        field_name = filter_item.get("field_name")
        value_target = filter_item.get("value_target")
        if field_name in required_fields or field_name in {"lang", "units"}:
            extracted_params.append({field_name: value_target})
    
    # *************** Ensure at least one valid input
    if not any(param for param in extracted_params if any(key in required_fields for key in param)):
        LOGGER.error("At least one of 'q', 'zip', 'lat', or 'lon' must be provided.")
    
    # *************** Call OpenWeather API for each valid input
    if intent_detected == "current_weather":
        url = "https://api.openweathermap.org/data/2.5/weather"
    elif intent_detected == "forecast":
        url = "https://api.openweathermap.org/data/2.5/forecast"
    responses = []
    for params in extracted_params:
        try:
            # *************** Add API key to parameters
            full_params = {"appid": SetupApi.weather_key, **params}
            response = requests.get(url, params=full_params)
            response.raise_for_status()
            responses.append(response.json())
        except requests.exceptions.RequestException as e:
            # *************** Log error if API call fails
            LOGGER.error(f"Error calling OpenWeather API: {e}")
            responses.append({"error": f"Error calling OpenWeather API: {e}"})
    
    return responses

# *************** Template prompt for response format weather
def prompt_response_format_weather() -> PromptTemplate:
    """
    Prompt to generate filter to use with weather api call

    Returns:
        PromptTemplate: _description_
    """
        
    template = """
    You are Professional Weather analyst. Your task is to convert the response from the weather mess information list dict into a readable format analysis information. 
    
    Input: 
    "response": {response}
    
    Instructions:
    1. Analyze the "response" from the weather mess information.
    2. Convert the response into a readable format analysis for weather information.
    
    Output:
    Return the response in a structured format.
    """
    
    return PromptTemplate(
        template=template,
        input_variables=["response"],
    )

# *************** Function to format the response from the OpenWeather API
def response_format_weather(response: List[Dict]): 
    """
    Call the OpenWeather API to get the current weather data for multiple locations.

    Args:
        response (dict): The response from the OpenWeather API call.

    Returns:
        list: A list of responses from the OpenWeather API (one for each input).
    """
    # *************** Validate the filters
    if not validate_list_input(response, 'response'):
        LOGGER.error("'response' must be a list.")
        return [{"error": "'response' must be a non-empty list."}]
    
    prompt = prompt_response_format_weather()
    
    # *************** Setup Runnable Chain
    runnale_filter = RunnableParallel(
                                {
                                    "response": itemgetter('response'),
                                  
                                }
                            )
    chain = ( 
             runnale_filter | 
             prompt | 
             LLM() | 
             StrOutputParser()
             )
    
    filter_response = chain.invoke(
                                    {
                                        "response": response
                                    }
                                )
    
    return filter_response

# *************** Function to handle current weather request
def handle_currrrent_weather(text_input:str, intent_detected: str, chat_history: list[dict]) -> str:
    """
    Handle the user's request for current weather information.

    Args:
        text_input (str): The user's input text.

    Returns:
        response_formatted: The formatted response for the current weather information.
    """
    
    # *************** Validate inputs text_input not an empty string
    if not validate_string_input(text_input, 'text_input'): 
        LOGGER.error("'text_input' must be a string.")
    
    filters = convert_text_to_filter(text_input, LIST_COLUMNS_FILTER, chat_history)
    weather_ouput = call_weather_api(filters, intent_detected)
    response_formatted = response_format_weather(weather_ouput)
    return response_formatted

# *************** Function to handle forecast request
def handle_forecast_weather(text_input:str, intent_detected: str, chat_history: list[dict]) -> str:
    """
    Handle the user's request for weather forecast information.

    Args:
        text_input (str): The user's input text.

    Returns:
        response_formatted: The formatted response for the weather forecast information
    """
    
    # *************** Validate inputs text_input not an empty string
    if not validate_string_input(text_input, 'text_input'):
        LOGGER.error("'text_input' must be a string.")
    
    filters = convert_text_to_filter(text_input, LIST_COLUMNS_FILTER, chat_history)
    weather_ouput = call_weather_api(filters, intent_detected)
    response_formatted = response_format_weather(weather_ouput)
    return response_formatted

# *************** Prompt to handle question unrelated to weather
def prompt_unrelated_question() -> PromptTemplate:
    """
    Prompt to handle unrelated user's input.
    
    Returns:
        PromptTemplate: The prompt template for generating the decision.
    """
    
    template = """ 
    You are a Weather expert assistant chatbot. 
    Your task is to respond professionally based on the 'input_text' and 'user_intent' with no greetings. 

    Input: 
    "input_text": {input_text}
    "user_intent": {user_intent}
    
    Instructions:
    1. Analyze the "input_text" and "user_intent".
    2. If the "input_text" is a general greeting, respond politely to assist the user.'
    3. If the "input_text" is unrelated to weather, respond professionally stating that the topic is out of scope, with no greetings or jargon technical terms.
    """
    
    return PromptTemplate(template=template, 
                          input_variable = ["input_text", "user_intent"])

# *************** Function to handle question unrelated to weather
def handle_unrelated_question(input_text:str, user_intent:str) -> str:
    """
    Generate response for unrelated user's input.

    Args:
        input_text (str): The input text to analyze.
        user_intent (str): The user's intent for the input text.

    Returns:
        str: The decision generated from the input text.
    """
    # *************** Validate inputs human_input not an empty string
    if not validate_string_input(input_text, 'input_text'):
        LOGGER.error("'input_text' must be a string.")
    if not validate_string_input(user_intent, 'user_intent'):
        LOGGER.error("'user_intent' must be a string.")
        
    template_prompt = prompt_unrelated_question()
    
    # *************** chaining prompt with llm 
    chain_chat = template_prompt | LLM() | StrOutputParser()
    result = chain_chat.invoke({"input_text": input_text, 
                                "user_intent": user_intent})
    return result

# *************** Prompt to handle incomplete filters user's input
def prompt_incomplete_filters() -> PromptTemplate:
    """
    Prompt to handle incomple filters user's input.
    
    return:
        PromptTemplate: The prompt template for generating the decision.
    """
    
    template = """
    You are a Weather expert assistant chatbot.
    Your task is to respond back to user's that one of the filters from "list_filters" 
    is missing in the "input_text".
    
    Inputs:
    "input_text": {input_text}
    "list_filters": {list_filters}
    
    Instructions:
    1. Analyze the "input_text" and "list_filters".
    2. Identify the missing filter from "list_filters" in the "input_text".
    3. Respond back to the user that the filter is missing. Don't use JARGON TECHNICAL TERMS.
    4. Provide a suggestion to the user to include the missing filter in the "input_text".
    """
    return PromptTemplate(template=template, 
                          input_variables=["input_text", "list_filters"])

# *************** Function to response to incomplete filters user's input
def handle_incomplete_filters(input_text:str, list_filters:dict) -> str:
    """
    Generate the response for incomplete filters in the user's input.

    Args:
        input_text (str): The input text to analyze.
        list_filters (dict): The list of filters to use with the OpenWeather API.

    Returns:
        str: The decision generated from the input text.
    """
    # *************** Validate inputs human_input not an empty string
    if not validate_string_input(input_text, 'input_text'):
        LOGGER.error("'input_text' must be a string.")
    if not validate_dict_input(list_filters, 'list_filters'):
        LOGGER.error("'list_filters' must be a dictionary.")
        
    template_prompt = prompt_incomplete_filters()
    
    # *************** chaining prompt with llm 
    chain_chat = template_prompt | LLM() | StrOutputParser()
    result = chain_chat.invoke({"input_text": input_text, 
                                "list_filters": list_filters})
    return result

# *************** Prompt for intent detection
def prompt_generate_decision() -> PromptTemplate:
    """
    Prompt to generate the decision based on the input text.

    Returns:
        PromptTemplate: The prompt template for generating the decision.
    """
    template = """
    You are a Weather expert assistant chatbot. 
    Your primary tasks: 
    - Detect the user's intent regarding weather information.
    - Classify the intent as one of the following categories:
      1. 'current_weather': If the user asks for current weather conditions or needs weather information (but if the user's have one of the filters in the "input_text").
      2. 'forecast': If the user asks for weather forecasts, or future conditions (but if the user's have one of the filters in the "input_text").
      3. 'historical_weather': If the user asks for past weather data (but if the user's have one of the filters in the "input_text").
      4. 'unknown': If the intent doesn't match the weather topic.
      5. 'incomplete': If the user's input is missing *all* required filters for a valid input based on "list_filters".
                          
    Note: 
     - Only *ONE* location filter is mandatory (e.g., 'city name' or 'zip' or 'lat/lon') based on "list_filters".
     - If the user doesn't provide optional fields (like 'units'), default to a suitable value (e.g., 'metric', 'imperial').
     - All other filters are OPTIONAL (e.g units, lang).  
     - Don't classified as incomplete intent if user's input "input_text" and previous message "chat_history" already have enough information at least one filter in "list_filters".
    
    ## Instructions:
      - If the user doesn't provide optional fields (like 'units'), default to a suitable value (e.g., 'metric').
      - Use "chat_history" to see if the user has already provided enough information in previous messages.
        - Look at "chat_history" type 'human' to see if the user has already provided enough information.
        
    Inputs:
      "input_text": {input_text}
      "list_filters": {list_filters}
      "chat_history": {chat_history}
    
    Output: 
    Response the output must followed this instructions. 
    {format_instructions}
    """
    
    return PromptTemplate(
        template=template,
        input_variables=["input_text", "list_filters", "chat_history"],
        partial_variables= {"format_instructions": JsonOutputParser(pydantic_object=IntentDetected).get_format_instructions()}
    )

# *************** Function to execute the decision intent
def generate_decision(input_text:str, list_filters:dict, chat_history: list[dict]) -> str:
    """
    Generate the decision based on the input text.

    Args:
        input_text (str): The input text to analyze.
        list_filters (dict): The list of filters to use with the OpenWeather API.
        chat_history (list): The chat history to use for context.

    Returns:
        str: The decision generated from the input text.
    """
    # *************** Validate inputs human_input not an empty string
    if not validate_string_input(input_text, 'input_text'):
        LOGGER.error("'input_text' must be a string.")
    
    formatted_hisotry = convert_chat_history(chat_history)
    context_window_history = context_window(formatted_hisotry)
    
    template_prompt = prompt_generate_decision()
    runnable_chain = RunnableParallel({
        "input_text": itemgetter('input_text'),
        "list_filters": itemgetter('list_filters'),
        "chat_history": itemgetter('chat_history')
    })
    # *************** chaining prompt with llm 
    chain_chat = runnable_chain| template_prompt | LLM() 
    result = chain_chat.invoke({"input_text": input_text,
                                "list_filters": list_filters,
                                "chat_history": context_window_history
                                })
    
    result = json_clean_output(result)
    return result

# *************** Main function to ask for weather information
def ask_to_chat(text_input: str, chat_history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Process chat input and generate appropriate weather-related responses.
    
    Args:
        text_input: User's input text
        chat_history: List of previous chat messages
    
    Returns:
        Tuple containing response information and updated chat history
    """
    try:
        if not validate_list_input(chat_history, 'chat_history', False):
            raise ValueError("Invalid chat history format")

        # *************** Generate intent
        intent_result = generate_decision(text_input, LIST_COLUMNS_FILTER, chat_history)
        print(f"\n\n intent result: {intent_result}")
        intent_detected = intent_result.get("intent_detected", [{}])[0].get("intent", "unknown")
        
        LOGGER.info(f"Detected intent: {intent_detected}")

        # *************** Intent handlers mapping
        intent_handlers = {
            "current_weather": handle_currrrent_weather,
            "forecast": handle_forecast_weather,
            "unknown": handle_unrelated_question,
            "incomplete": lambda x, y: handle_incomplete_filters(x, LIST_COLUMNS_FILTER)
        }

        # *************** Get handler and process response
        handler = intent_handlers.get(intent_detected, handle_unrelated_question)
        if intent_detected in {"current_weather", "forecast"}:
            response_information = handler(text_input, intent_detected, chat_history)
        else:
            response_information = handler(text_input, intent_detected)

        # *************** Update history
        history = save_chat_history(chat_history, text_input, response_information)
        return response_information, history

    except Exception as e:
        LOGGER.error(f"Error processing chat: {str(e)}")



if __name__ == "__main__":
    payload = { 
               "text_input": "i need for city yogyakarta",
               "chat_history": [{'type': 'human', 'content': 'give me weather information'}, {'type': 'ai', 'content': 'It looks like your request for weather information is missing some important details. You need to specify a location using one of the following filters: a city name, a ZIP or postal code, or coordinates (latitude and longitude). \n\nPlease include one of these details in your request so I can help you better!'}]
               }
    filters, history = ask_to_chat(payload['text_input'], payload['chat_history'])
    print(filters)
    print(f"\n\n {history}")