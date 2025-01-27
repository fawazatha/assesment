
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from typing import Dict, Tuple

def process_history_entry(entry: Dict | BaseMessage) -> Tuple[str, str]:
    """Process a single history entry"""
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
    
    # *************** Initialize context window
    context_window = []
    
    # *************** Add messages to context window
    for i in range(0, len(chat_history)):
        if i + 1 >= len(chat_history):
            break 
        
        human_msg, ai_msg = process_history_entry(chat_history[i])
        context_window.extend([
            HumanMessage(content=human_msg),
            AIMessage(content=ai_msg)
        ])
    
    # *************** Return context window
    window_start = max(0, len(context_window) - (window_size * 2))
    return chat_history[window_start:]


if __name__ == "__main__":
    # *************** Test context window
    chat_history = [
    {"human": "Hi there!", "ai": "Hello! How can I assist you?"},
    {"human": "Can you help me with Python?", "ai": "Of course! What do you need help with?"},
    {"human": "I need help with writing a function.", "ai": "Sure, what function are you trying to write?"},
    {"human": "A function to calculate factorial.", "ai": "Okay! I can show you an example of that."},
]
    context_window = context_window(chat_history)
    for message in context_window:
        print(message)