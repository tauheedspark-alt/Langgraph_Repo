!pip install langgraph
!pip install langchain
!pip install typing

from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class BMIState(TypedDict) :
  weight_kg : float
  height_m : float
  bmi : float
  category : str

def bmi_calculator(state: BMIState) -> BMIState:
    height = state["height_m"]
    weight = state["weight_kg"]

    bmi = weight / (height ** 2)
    bmi = round(bmi, 2)

    state["bmi"] = bmi
    return state

def label_bmi(state: BMIState) -> BMIState:

    bmi = state['bmi']

    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"

    return state

#create graph
graph = StateGraph(BMIState)
#add node
graph.add_node('bmi_calcualtor',bmi_calculator)
graph.add_node('label_bmi',label_bmi)
# add edge
graph.add_edge(START,'bmi_calcualtor')
graph.add_edge('bmi_calcualtor','label_bmi')
graph.add_edge('label_bmi',END)

workflow =graph.compile()

# execute the graph
intial_state = {'weight_kg':65, 'height_m':1.73}

final_state = workflow.invoke(intial_state)

print(final_state)

