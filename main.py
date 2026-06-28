from typing import TypedDict, List,Literal,Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

class AuditorFeedback(BaseModel):
    is_compliant: bool
    missing_specs: List[str]
    hallucinations_detected: List[str]
    revision_instructions: str

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
    status: Literal["APPROVED_BY_AI","ESCALATED_TO_HUMAN"] # "APPROVED_BY_AI" or "ESCALATED_TO_HUMAN"

def extractor(state:State):
    print("Extracting key variables...")
    print("Key variables extracted!\n\n")
    return{
        "the_goal":"Needs 180C heat resistance and needs 10% tint"
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
    return {
        "current_draft":"Dear Consultant, our Window-Model-X exceeds your heat requirement of 180°C because it is rated for 200°C. It also meets your shading needs with a 12% tint."
    }

def auditor(state:State):
    print("Checking and comparing datasheet...")
    print("All looks good!\n\n")
    return{
        "final_compliance_statement":state["current_draft"],
        "status":"APPROVED_BY_AI"
    }

graph_builder = StateGraph(State)
graph_builder.add_node("extractor",extractor)
graph_builder.add_node("retriever",retriever)
graph_builder.add_node("writer",writer)
graph_builder.add_node("auditor",auditor)

graph_builder.add_edge(START,"extractor")
graph_builder.add_edge("extractor","retriever")
graph_builder.add_edge("retriever","writer")
graph_builder.add_edge("writer","auditor")
graph_builder.add_edge("auditor",END)

graph = graph_builder.compile()

if __name__ == "__main__":
    initial_state = {
        "requirements_text":"",
        "product_sku":""
    }
    
    response = graph.invoke(initial_state)
    print(f"Got response from the graph!\nResult:{response}")