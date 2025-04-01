from typing import Optional
import re
import json
from wasabi import msg

def extract_json(text: str) -> dict:
    """
    Extract JSON from a string.

    Parameters:
    -----------
    text: str
        The string to extract JSON from.

    Returns:
    --------
    data: dict
        The extracted JSON data.
    """
    if "```json" in text:
        text = re.sub(r"```json", "", text)  # 移除 ```json
    if "```" in text:
        text = re.sub(r"```", "", text)

    text = text.replace(r"\n", "\n")  # 还原 \n 为换行符

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
    
def print_conversation(msgs):
    """
    Print the conversation in a readable format.

    Parameters:
    -----------
    msg: list
        The conversation messages to print.
    """
    for turn in msgs:
        icon = "🤖" if turn['role'] == "assistant" else (
            "⚙️" if turn['role'] == "system" else "👤")
        msg.divider(icon)
        print(turn['content'])