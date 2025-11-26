import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import json

load_dotenv()

SERVERS = { 
    "math": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "--directory",
            "D:/Learning2/LLMs/MCP/campusx/math-mcp-server",
            "run",
            "main.py"
       ]
    },
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": "https://mayank-vikki.fastmcp.app/mcp"
    },
    "manim-server": {
        "transport": "stdio",
        "command": "C:/Users/Mayank/anaconda3/python.exe",
        "args": [
        "D:/Learning2/LLMs/MCP/campusx/manim-mcp-server-fixed/manim_server.py"
      ],
        "env": {
        "MANIM_EXECUTABLE": "C:/Users/Mayank/anaconda3/Scripts/manim.exe"
      }
    }
}

async def main():
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    # print("Available tools:", [tool.name for tool in tools])
    # print("\n" + "="*80)
    # print("TOOL DEFINITIONS:")
    # print("="*80)
    
    # for tool in tools:
    #     print(f"\nTool: {tool.name}")
    #     print(f"Description: {tool.description}")
    #     print(f"Args Schema: {json.dumps(tool.args_schema, indent=2)}")
    #     print("-"*80)

    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    # print("\nAvailable tools:", named_tools.keys())

    llm = ChatOpenAI(model="gpt-5")
    llm_with_tools = llm.bind_tools(tools)

    prompt = "Draw a triangle rotating in place using the manim tool."
    #prompt = "Add 880 Rs to education on 4th Nov'25"
    print(f"\nSending prompt to LLM: {prompt}")
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return
    
    print(f"\nLLM wants to call {len(response.tool_calls)} tool(s)...")
    
    
    # selected_tool = response.tool_calls[0]["name"]
    # selected_tool_args = response.tool_calls[0].get("args") or {}
    # selected_tool_id = response.tool_calls[0]["id"]
    # result = await named_tools[selected_tool].ainvoke(selected_tool_args)
    # tool_message = ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result))
    # final_response = await llm_with_tools.ainvoke([prompt, response, tool_message])
    # print(f"Final response: {final_response.content}")
   

   

    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        print(f"\nExecuting tool: {selected_tool}")
        print(f"With args: {json.dumps(selected_tool_args, indent=2)}")
        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        print(f"Tool execution complete!")
        print(f"Result: {json.dumps(result, indent=2)}")
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
        

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print(f"Final response: {final_response.content}")


if __name__ == '__main__':
    asyncio.run(main())