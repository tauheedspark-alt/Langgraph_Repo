from typing import TypedDict, Literal
from google.colab import userdata
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph.state import Checkpoint
api_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0
)
model = ChatHuggingFace(llm=llm)

class ChatState(TypedDict):
    chathistory : Annotated[list[BaseMessage], add_messages]

def chatboat(state : ChatState):
  message = state['chathistory']
  result = model.invoke(message).content

  return {'chathistory': [result]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chatbot",chatboat)

graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)

workflow=graph.compile(checkpointer=checkpointer)

thread_id =1
while True :
  user_message =input("type here")

  if user_message.strip().lower() in ["quit","exit","bye"]:
    break

  config = {'configurable' : {'thread_id': thread_id}}
  human_message =HumanMessage(content=user_message)
  state ={'chathistory' : human_message}
  response = workflow.invoke(state,config=config)
  print(response['chathistory'][-1].content)

