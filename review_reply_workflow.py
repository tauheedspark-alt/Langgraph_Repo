!pip install langchain-huggingface
from typing import TypedDict, Literal
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from google.colab import userdata
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class SentimentSchema(BaseModel):
  sentiment : Literal['positive','negative'] =  Field(description='Sentiment of the review')

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

class reviewState(TypedDict):
  review : str
  sentiment : Literal['positive','negative']
  diagnosis: dict
  response : str

api_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0
)

model = ChatHuggingFace(llm=llm)


parser1 = PydanticOutputParser(pydantic_object=SentimentSchema)
parser2 = PydanticOutputParser(pydantic_object=DiagnosisSchema)

def find_sentiment(state:reviewState) :
  review = state['review']
  prompt = PromptTemplate(
        template="""For the following review find out the sentiment
        {format_instructions}
        {review}
        """,
        input_variables=["reivew"],
        partial_variables={
            "format_instructions": parser1.get_format_instructions()
        }
    )
  prompt = prompt.invoke({"review": review})
  result = model.invoke(prompt)
  parsed_result = parser1.parse(result.content)
  return {'sentiment':parsed_result.sentiment}

def run_diagonsis(state : reviewState):
  review =state['review']
  prompt = PromptTemplate(
        template="""Diagnose this negative review:\n\n
        {format_instructions}
        {review} \n
        Return issue_type, tone, and urgency
        """,
        input_variables=["reivew"],
        partial_variables={
            "format_instructions": parser2.get_format_instructions()
        }
    )
  prompt = prompt.invoke({"review": review})
  result = model.invoke(prompt)
  parsed_result = parser2.parse(result.content)
  return {'diagnosis': parsed_result.model_dump()}

def check_sentiment(state: reviewState) -> Literal["positive_response", "run_diagonsis"]:

    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagonsis'


def negative_response(state: reviewState):
  diagnosis = state['diagnosis']
  prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
  result = model.invoke(prompt)
  return {'response':result.content}


def positive_response(state: reviewState):
  review =state['review']
  prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{review}\"\n
Also, kindly ask the user to leave feedback on our website."""
  response = model.invoke(prompt).content
  return {'response': response}

from langgraph.graph import StateGraph,START,END

graph = StateGraph(reviewState)

graph.add_node("find_sentiment",find_sentiment)
graph.add_node("positive_response",positive_response)
graph.add_node("run_diagonsis",run_diagonsis)
graph.add_node("negative_response",negative_response)


#both format 1 and format 2 are same

#format 1
#graph.add_edge(START,"find_sentiment")
#graph.add_conditional_edges("find_sentiment",check_sentiment)
#graph.add_edge('positive_response', END)
#graph.add_edge('run_diagonsis', 'negative_response')
#graph.add_edge('negative_response', END)

#format 2
graph.add_edge(START,"find_sentiment")
graph.add_conditional_edges("find_sentiment",check_sentiment, {'positive_response' : 'positive_response' , 'run_diagonsis':'run_diagonsis'})
graph.add_edge('positive_response', END)
graph.add_edge('run_diagonsis', 'negative_response')
graph.add_edge('negative_response', END)


workflow =graph.compile()

intial_state={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
workflow.invoke(intial_state)


