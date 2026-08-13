

!pip install langchain-huggingface
essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""

essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India"."""

class UPSCEASSY(BaseModel):
  feedback : str = Field(description='Detailed feedbackfor the essay')
  score : int =Field(description='Score out of 10' ,ge=0,le=10)

from google.colab import userdata
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import TypedDict,operator,Annotated


api_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0
)

model = ChatHuggingFace(llm=llm)


parser = PydanticOutputParser(pydantic_object=UPSCEASSY)

class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float
    final_result : str

def evaluate_language_feedback(state: UPSCState):
  essay = state["essay"]

  prompt = PromptTemplate(
        template="""
        Evaluate the language quality of the following essay
        and provide feedback and assign a score out of 10.

        {format_instructions}

        Essay:
        {essay}
        """,
        input_variables=["essay"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

  prompt = prompt.invoke({"essay": essay})
  result = model.invoke(prompt)
  parsed_result = parser.parse(result.content)
  print(parsed_result.score)
  return {'language_feedback' : parsed_result.feedback, 'individual_scores' : [parsed_result.score]}

def evaluate_analysis_feedback(state: UPSCState):
  essay = state["essay"]

  prompt = PromptTemplate(
        template="""
        Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 .
        {format_instructions}

        Essay:
        {essay}
        """,
        input_variables=["essay"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

  prompt = prompt.invoke({"essay": essay})
  result = model.invoke(prompt)
  parsed_result = parser.parse(result.content)
  print(parsed_result.score)
  return {'analysis_feedback' : parsed_result.feedback, 'individual_scores' : [parsed_result.score]}

def evaluate_clarity_feedback(state: UPSCState):
  essay = state["essay"]

  prompt = PromptTemplate(
        template="""
        Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10  .
        {format_instructions}

        Essay:
        {essay}
        """,
        input_variables=["essay"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

  prompt = prompt.invoke({"essay": essay})
  result = model.invoke(prompt)
  parsed_result = parser.parse(result.content)
  print(parsed_result.score)
  return {'clarity_feedback' : parsed_result.feedback, 'individual_scores' : [parsed_result.score]}

def evaluate_overall_feedback(state: UPSCState):
  essay = state["essay"]
  language_feedback = state["language_feedback"]
  analysis_feedback = state["analysis_feedback"]
  clarity_feedback = state["clarity_feedback"]
  prompt =f'Based on the following feedbacks create a summarized feedback \n language feedback - {language_feedback} \n depth of analysis feedback - {analysis_feedback} \n clarity of thought feedback - {clarity_feedback}'
  # avg calculate
  overall_feedback = model.invoke(prompt).content  
  print(state['individual_scores'])
  avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

  return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

def final_Result(state: UPSCState):
  final_score =state['avg_score']
  if final_score >7.5 :
      final_result = "Congurlation you are selected for UPSC Interview process.."
  else :
      final_result = "Sorry Not selected..."
  return {'final_result':final_result}     

from langgraph.graph import StateGraph,START,END
graph = StateGraph(UPSCState)

# add node

graph.add_node("evaluate_language_feedback",evaluate_language_feedback)
graph.add_node("evaluate_analysis_feedback",evaluate_analysis_feedback)
graph.add_node("evaluate_clarity_feedback",evaluate_clarity_feedback)
graph.add_node("evaluate_overall_feedback",evaluate_overall_feedback)
graph.add_node("final_Result",final_Result)


# add edge

graph.add_edge(START,"evaluate_language_feedback")
graph.add_edge(START,"evaluate_analysis_feedback")
graph.add_edge(START,"evaluate_clarity_feedback")
graph.add_edge("evaluate_language_feedback","evaluate_overall_feedback")
graph.add_edge("evaluate_analysis_feedback","evaluate_overall_feedback")
graph.add_edge("evaluate_clarity_feedback","evaluate_overall_feedback")
graph.add_edge("evaluate_overall_feedback","final_Result")
graph.add_edge("final_Result",END)


# compile graph
workflow =graph.compile()

initial_state = {'essay':essay}
result = workflow.invoke(initial_state)
print(result["final_result"])

