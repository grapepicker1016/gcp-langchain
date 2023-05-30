from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

def conversation_agent(input: str) -> str :    # first initialize the large language model
    llm = OpenAI(
        temperature=0,
        model_name="text-davinci-003",
    )

    # now initialize the conversation chain
    conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    result = conversation(input)
    print({result})
    print({conversation})
    return(result)
