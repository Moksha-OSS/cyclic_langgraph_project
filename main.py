from typing import TypedDict, List,Literal,Annotated
from pydantic import BaseModel,Field
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

class AuditorFeedback(BaseModel):
    is_compliant: Literal[True,False] = Field(
        description="is the product compilant with what the customer asked answer in 'True' or 'False'"
    )
    missing_specs: List[str] = Field(
        description="make a list of all the missing specs that our prodcut does have that are required in by the user"
    )
    hallucinations_detected: List[str] = Field(
        description="make a list of all the hallucinations that the current draft holds comparing the users requirements and the spec sheet for our product"
    )
    revision_instructions: str = Field(
        description="Any revisions instructions that you would like to give to the drafter so that it can make better draft"
    )

class ExtractedGoal(BaseModel):
    goal:str = Field(
        description="What are main things that the user want for example 'The product needs to withstand 180C heat and needs 10%tint'"
    )

class Draft(BaseModel):
    drafted_response:str = Field(
        description="According to the data given to you i.e the user's requirements and what our product has draft a response for the user stating how the product we have satisfies and meets their product requirements"
    )

class State(TypedDict):
    # Inputs
    requirements_text: str
    product_sku: str
    
    the_goal:str

    # RAG Context
    retrieved_datasheet_specs: str
    
    # Iteration Memory
    current_draft: str
    feedback_history: Annotated[List[AuditorFeedback],add_messages]
    iteration_count: int
    
    # Final Output
    final_compliance_statement: str
    status: Literal["APPROVED_BY_AI","ESCALATED_TO_HUMAN"]

llm = ChatGoogleGenerativeAI(model = "google-3.1-flash-lite",temperature=0)
def extractor(state:State):
    print("Extracting key variables...")
    print("Key variables extracted!\n\n")
    llm_with_structured_output = llm.with_structured_output(ExtractedGoal)
    system_message = SystemMessage(content="""""")
    message_for_ai = [SystemMessage,HumanMessage(content=state['requirements_text'])]
    response = llm_with_structured_output.invoke(message_for_ai)
    return{
        "the_goal":response.goal
    }

def retriever(state:State):
    print("Going into database...")
    print("Data extracted!\n\n")
    return {
        "retrieved_datasheet_specs":"Window-Model-X can survive 200C and has a 12% tint"
    }

def writer(state:State):
    print("Drafting a response...")
    print("Response Drafted!\n\n")
    llm_with_structured_output = llm.with_structured_output(Draft)
    system_message = SystemMessage(content="""""")
    human_message = HumanMessage(content=f"""
Here are the requirements that user wants

{state['requirements_text']}

and here are the products that we have in our inventory and that are closes to satisfying user needs

{state['retrieved_datasheet_specs']}
""")
    message_for_llm = [system_message,human_message]
    response = llm_with_structured_output.invoke(message_for_llm)
    return {
        "current_draft":response.drafted_responose
    }

def auditor(state:State):
    print("Checking and comparing datasheet...")
    print("All looks good!\n\n")
    llm_with_structured_output = llm.with_structured_output(AuditorFeedback)
    system_message = SystemMessage(content="""""")
    human_message = HumanMessage(content=f"""{state['requirements_text']}\n{state['retrieved_datasheet_specs']}\n{state['current_draft']}""")
    response = llm_with_structured_output.invoke([system_message,human_message])
    if not response.is_compliant:
        state["iteration_count"]+=1
        return{
            "feedback_history":[response]
        }
    return{
        "final_compliance_statement":state["current_draft"],
        "status":"APPROVED_BY_AI"
    }

def classifier(state:State):
    if state["iteration_count"] >=3:
        state["status"] = "ESCALATED_TO_HUMAN"
        return END
    elif state["iteration_count"] < 3 and state["feedback_history"][-1].is_compliant == False:
        return "writer"

graph_builder = StateGraph(State)
graph_builder.add_node("extractor",extractor)
graph_builder.add_node("retriever",retriever)
graph_builder.add_node("writer",writer)
graph_builder.add_node("auditor",auditor)

graph_builder.add_edge(START,"extractor")
graph_builder.add_edge("extractor","retriever")
graph_builder.add_edge("retriever","writer")
graph_builder.add_edge("writer","auditor")
graph_builder.add_conditional_edges("auditor",classifier)

graph = graph_builder.compile()

if __name__ == "__main__":
    initial_state = {
        "requirements_text":"",
        "product_sku":""
    }
    
    response = graph.invoke(initial_state)
    print(f"Got response from the graph!\nResult:{response}")