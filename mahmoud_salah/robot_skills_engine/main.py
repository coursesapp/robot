import json
from duckduckgo_search import DDGS
# --- 1. The Actual Python Functions (The Tools) ---

def execute_python_code(code_string):
    print(f"Executing code:\n{code_string}")
    return "Code executed successfully."

def control_hardware_motors(body_part, action):
    # In the future, this is where you connect to the physical servos
    print(f"[HARDWARE ALERT] Moving {body_part} to perform action: {action}")
    return f"Successfully moved {body_part}."

def get_current_weather(location):
    print(f"Fetching weather data for {location}...")
    return "It is currently 24 degrees and sunny."

def perform_deep_search(query):
    print(f"[WEB SEARCH] Looking up: '{query}'...")
    try:
        # Initialize the DuckDuckGo search
        with DDGS() as ddgs:
            # Get the top 3 text results from the internet
            results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return "No results found on the web."
            
            # Format the results so the LLM can easily read them
            search_summary = "Here are the top web results:\n\n"
            for i, r in enumerate(results):
                search_summary += f"Result {i+1} Title: {r['title']}\nSnippet: {r['body']}\n\n"
            
            return search_summary
            
    except Exception as e:
        return f"Error connecting to the internet: {e}"

# --- 2. The Tool Router ---

def execute_agent_tool(llm_json_output):
    """
    This function takes the JSON output from the LLM and routes it
    to the correct Python function.
    """
    print("--- Incoming Command from AI Brain ---")
    
    # Extract the tool name and arguments from the LLM's JSON
    tool_name = llm_json_output["tool_choice"]
    arguments = llm_json_output["arguments"]
    
    # Route to the correct function based on the tool name
    if tool_name == "execute_python_code":
        result = execute_python_code(arguments["code_string"])
        
    elif tool_name == "control_hardware_motors":
        result = control_hardware_motors(arguments["body_part"], arguments["action"])
        
    elif tool_name == "get_current_weather":
        result = get_current_weather(arguments["location"])
        
    elif tool_name == "perform_deep_search":
        result = perform_deep_search(arguments["query"])
        
    else:
        result = "Error: Tool not found."
    
      
    print(f"Result sent back to AI: {result}\n")
    return result

# --- 3. Simulation for your Submission ---

if __name__ == "__main__":
    # Pretend the user said: "Raise your right arm!"
    # This is the JSON the LLM would output based on our tools_config.json
    mock_llm_response = {
        "tool_choice": "control_hardware_motors",
        "arguments": {
            "body_part": "right_arm",
            "action": "raise"
        }
    }
    
    # Run the router
    execute_agent_tool(mock_llm_response)