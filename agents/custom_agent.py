from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from os import getenv
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from typing import Optional
import os
import json
from pipedrive.client import Client

load_dotenv()

client = Client(domain="https://trufaarte.pipedrive.com/")
PIPEDRIVE_API_KEY = os.getenv("PIPEDRIVE_API_KEY")
client.set_api_token(PIPEDRIVE_API_KEY)
print(f"PIPEDRIVE_API_KEY: {PIPEDRIVE_API_KEY}")


def get_person_details(phone: str) -> str:
    """Returns the person details using the phone number."""
    # phone = phone or "5580496309"
    # print(f"phone number: {phone}")

    if "=" in phone:
        phone = phone.split("=")[1]

    print(f"phone number 2: {phone}")
    result = client.persons.search_persons(params={"term": phone, "fields": "phone"})

    print(f"result: {result}")

    return json.dumps(result)


get_person_details_tool = StructuredTool.from_function(get_person_details)

load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")


tools = [
    Tool(
        name="get_person_details",
        func=lambda phone: get_person_details_tool(phone),
        description="This tool is used to get the person details providing a phone number argument. Use this tool when the personal details are not available.",
    )
]

llm = ChatOpenAI

template = """Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action as required by the tool, without any extra text
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:

Begin!

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

agent_executor.run("Get the person details from the CRM for phone number 5580496309")
