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

    # Response with tool_calls
    response_with_tool = completion(
        model='gpt-5.4-mini',
        messages=[{"role": "user", "content": "What is 1234 x 5678?"}],
        tools=tools,
    )

    print("\n# Multiplication question needs a tool call")
    print(response_with_tool.choices[0].message.content)
    print(response_with_tool.choices[0].message.tool_calls)

    # Feed result back to LLM
    ai_message = response_with_tool.choices[0].message
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "calculator":
                result = calculator(**function_args)
                print(f"\n{function_name}({function_args}) -> {result}")

    # We append the assistant's tool-call message, the tool role result,
    # and let the model produce a final answer.
    messages = [{"role": "user", "content": "What is 1234 x 5678?"}]
    ai_message = response_with_tool.choices[0].message

    # A: append the assistant's tool call
    messages.append({
        "role": "assistant",
        "content": ai_message.content,
        "tool_calls": ai_message.tool_calls,
    })

    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            function_name = tool_call.function.name  # B: parse + execute the tool
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "calculator":
                result = calculator(**function_args)
                messages.append({                    # C: append the tool result
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

    final_response = completion(
        model='gpt-5.4-mini',
        messages=messages,
    )

    print("\nMessages:", messages)
    print("\nFinal Answer:", final_response.choices[0].message.content)


# ---------------------------------------------------------------------
# End of File
# ---------------------------------------------------------------------
