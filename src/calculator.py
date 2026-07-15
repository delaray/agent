# Setup
import json
from litellm import completion
from dotenv import load_dotenv

load_dotenv(override=True)


# ---------------------------------------------------------------------

calculator_tool_definition = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "description": "Arithmetic operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "first_number": {
                    "type": "number",
                    "description": "First number for the calculation"
                },
                "second_number": {
                    "type": "number",
                    "description": "Second number for the calculation"
                }
            },
            "required": ["operator", "first_number", "second_number"],
        }
    }
}


# ---------------------------------------------------------------------

def calculator(operator: str, first_number: float, second_number: float):
    if operator == 'add':
        return first_number + second_number
    elif operator == 'subtract':
        return first_number - second_number
    elif operator == 'multiply':
        return first_number * second_number
    elif operator == 'divide':
        if second_number == 0:
            raise ValueError("Cannot divide by zero")
        return first_number / second_number
    else:
        raise ValueError(f"Unsupported operator: {operator}")


# ---------------------------------------------------------------------

def example1():
    tools = [calculator_tool_definition]

    response_without_tool = completion(
        model='gpt-5.4-mini',
        messages=[
            {"role": "user", "content": "What is the capital of South Korea?"}],
        tools=tools,
    )

    print("# Capital question doesn't need a tool call")
    print(response_without_tool.choices[0].message.content)
    print(response_without_tool.choices[0].message.tool_calls)

    response_with_tool = completion(
        model='gpt-5.4-mini',
        messages=[{"role": "user", "content": "What is 1234 x 5678?"}],
        tools=tools,
    )

    print("\n# Multiplication question needs a tool call")
    print(response_with_tool.choices[0].message.content)
    print(response_with_tool.choices[0].message.tool_calls)

# ---------------------------------------------------------------------
# End of File
# ---------------------------------------------------------------------
