import os
import json

from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.tools.json.tool import JsonSpec
from langchain.tools import StructuredTool
import os
from pipedrive.client import Client

client = Client(domain='https://trufaarte.pipedrive.com/')
PIPEDRIVE_API_KEY=os.getenv('PIPEDRIVE_API_KEY')
client.set_api_token(PIPEDRIVE_API_KEY)

def get_contact(body: dict) -> str:
    """Uses Pipedrive client to get the person object from the CRM using the person {id:int}."""
    id = int(body.id) or 45
    
    result = client.persons.get_person(id)
    
    return result

data= None
json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0),
    toolkit=json_toolkit,
    verbose=True
)