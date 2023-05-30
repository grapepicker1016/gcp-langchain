from langchain.tools import StructuredTool
from typing import Optional
import os
import json
from pipedrive.client import Client

client = Client(domain='https://trufaarte.pipedrive.com/')
PIPEDRIVE_API_KEY=os.getenv('PIPEDRIVE_API_KEY')
client.set_api_token(PIPEDRIVE_API_KEY)

def get_contact(phone:str) -> str:
    """Uses Pipedrive client to search the person in the CRM using the phone number."""
    phone = phone or "5580496309"
    
    result = client.persons.search_persons(phone)
    
    return json.dumps(result)

# get_contact_tool = StructuredTool.from_function(get_contact)