# Role
You are a talented Python deveoper with excdvellent GUI design skills.

## Context
I have built a basic agent module that includes components and protocols for
exceution context, tool definition and tool use, an LLM communication layer
and an agent loop that wraps it all together. I plan on augmenting eventually
and step by step agentmenting the agent with RAG-based knowledge bases, memory
management features, planning and reflection and code execution mechanisms.

# Task Overview
I would like you to build a extensible GUI that will allow me to on the one hand use
the CLI functionality interactively by selecting providers and models, specifying
model parameters, and tool selection. I would like a trace facility that shows the details
of each evernt including parameter values and tool result values.

# Task Details
Your task is to build a GUI for the agent module. The GUI should serve several purposes:
1. Inputting a task or query and displaying the results
2. Monitoring the task accomplishment process by showing individual tool use and thinking steps

# Guidelines
1. Think and deign the GUI
2. Implement the GUI in Streamlit
3. Leverage existing funcftionality in src
4. Add utilities and helper functions to utils.py as needed
5. Make the GUI professional looking and intuitive to use
6. Be creative and your design intuitions.

# -----------------------------------------------------------------------------

# First Bug Fix
It looks great. The only thing that needs changing is the foreground color of the whitish text. It needs to be brighter as it is really hard to see/read even on a very darkl background.
