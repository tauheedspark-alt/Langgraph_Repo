# Developer : Tauheed Ahmad
# version : 1.0
!pip install langgraph
!pip install langchain
!pip install typing
!pip install -U langchain-huggingface huggingface_hub
!pip install langchain-huggingface
!pip install sentence-transformers
!pip install -U langchain langchain-core langchain-huggingface huggingface_hub

# creating text generatin open source llm

from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from google.colab import userdata
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

import os

api_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["USER_AGENT"] = "Mozilla/5.0"
llm =HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0
)
model = ChatHuggingFace(llm = llm)

#state class for langgraph

class LLMSTATE(TypedDict) :
  query : str
  outline : str
  blog : str


def generate_outline(state : LLMSTATE)  -> LLMSTATE:
  query =state['query']
  prompt = f'please genereate the details outline for given context {query}'
  outline = model.invoke(prompt).content
  state['outline'] = outline
  return state

def generate_blog(state : LLMSTATE)  -> LLMSTATE:
  outline =state['outline']
  prompt = f'please genereate the details blog for given outline {outline}'
  blog = model.invoke(prompt).content
  state['blog'] = blog
  return state

# create graph
graph = StateGraph(LLMSTATE)


# add nodes
graph.add_node('generate_outline',generate_outline)
graph.add_node('generate_blog',generate_blog)

# add edge
graph.add_edge(START,'generate_outline')
graph.add_edge('generate_outline','generate_blog')
graph.add_edge('generate_blog',END)

# compile graph
workflow = graph.compile()
intial_state = {'query' : 'evaluation of AI in india'}
result = workflow.invoke(intial_state)
print(result['outline'])
print(result['blog'])




