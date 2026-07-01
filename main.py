from typing import TypedDict, List, Literal, Annotated
import operator
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# 1. DATA MODELS (PYDANTIC)
# ---------------------------------------------------------
class AuditorFeedback(BaseModel):
    is_compliant: Literal[True, False] = Field(
        description="True if the draft is 100% compliant with the requirements based ONLY on the datasheet. False if there are hallucinations or missing specs."
    )
    missing_specs: List[str] = Field(
        description="List of client requirements that were not addressed in the draft."
    )
    hallucinations_detected: List[str] = Field(
        description="List of claims in the draft that contradict or are not supported by the datasheet."
    )
    revision_instructions: str = Field(
        description="Clear, direct instructions for the writer on how to fix the draft in the next iteration."
    )

class ExtractedGoal(BaseModel):
    goal: str = Field(
        description="A concise summary of all technical requirements extracted from the client's request."
    )

class Draft(BaseModel):
    drafted_response: str = Field(
        description="The formal compliance statement letter addressed to the consultant."
    )

# ---------------------------------------------------------
# 2. STATE GRAPH DEFINITION
# ---------------------------------------------------------
class State(TypedDict):
    # Inputs
    requirements_text: str
    product_sku: str
    the_goal: str

    # RAG Context
    retrieved_datasheet_specs: str
    
    # Iteration Memory
    current_draft: str
    feedback_history: Annotated[List[AuditorFeedback], operator.add]
    iteration_count: int
    
    # Final Output
    final_compliance_statement: str
    status: Literal["APPROVED_BY_AI", "ESCALATED_TO_HUMAN", "REJECTED", "PENDING"]

# ---------------------------------------------------------
# 3. AI INITIALIZATION
# ---------------------------------------------------------
# Ensure you have your GOOGLE_API_KEY set in your environment variables
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# ---------------------------------------------------------
# 4. GRAPH NODES
# ---------------------------------------------------------
def extractor(state: State):
    print("--- [NODE: EXTRACTOR] ---")
    llm_with_structured_output = llm.with_structured_output(ExtractedGoal)
    
    system_message = SystemMessage(
        content="You are an expert engineering data extractor. Read the client's requirement text and summarize the core technical requirements into a single, highly accurate checklist."
    )
    human_message = HumanMessage(content=state['requirements_text'])
    
    response = llm_with_structured_output.invoke([system_message, human_message])
    print(f"Goal Extracted: {response.goal}\n")
    
    return {"the_goal": response.goal}


def retriever(state: State):
    print("--- [NODE: RETRIEVER] ---")
    # In a real app, this searches a pgvector database using the product_sku and the_goal.
    # For now, we pass the test data through.
    print(f"Datasheet fetched for {state['product_sku']}\n")
    return {"retrieved_datasheet_specs": state["retrieved_datasheet_specs"]}


def writer(state: State):
    print(f"--- [NODE: WRITER (Iteration {state['iteration_count']})] ---")
    llm_with_structured_output = llm.with_structured_output(Draft)
    
    system_message = SystemMessage(
        content="""You are a technical compliance writer for a construction materials supplier. 
        Draft a formal, professional statement explaining how our product meets the client's requirements. 
        Rule 1: NEVER invent or hallucinate specifications. Rely ONLY on the provided datasheet.
        Rule 2: If a requirement cannot be met, state clearly what our product actually provides instead."""
    )
    
    # Check if we have previous feedback to learn from
    feedback_context = ""
    if state["iteration_count"] > 0 and len(state["feedback_history"]) > 0:
        latest_feedback = state["feedback_history"][-1]
        feedback_context = f"\n\nURGENT FEEDBACK FROM AUDITOR ON PREVIOUS DRAFT:\nInstructions: {latest_feedback.revision_instructions}\nHallucinations to fix: {latest_feedback.hallucinations_detected}"

    human_message = HumanMessage(content=f"""
    Client Requirements: {state['requirements_text']}
    
    Official Product Datasheet: {state['retrieved_datasheet_specs']}
    {feedback_context}
    """)
    
    response = llm_with_structured_output.invoke([system_message, human_message])
    print("New draft generated.\n")
    
    return {"current_draft": response.drafted_response}


def auditor(state: State):
    print("--- [NODE: AUDITOR] ---")
    llm_with_structured_output = llm.with_structured_output(AuditorFeedback)
    
    system_message = SystemMessage(
        content="""You are a strict QA Engineering Auditor. Compare the drafted response against the client requirements and the official product datasheet.
        If the draft claims a specification that is not supported by the datasheet, mark is_compliant as False and list the hallucination.
        If the draft claims it meets a requirement that the datasheet proves it fails, mark is_compliant as False.
        Be merciless. Engineering safety depends on your accuracy."""
    )
    
    human_message = HumanMessage(content=f"""
    Client Requirements: {state['requirements_text']}
    Official Datasheet: {state['retrieved_datasheet_specs']}
    Current Draft: {state['current_draft']}
    """)
    
    response = llm_with_structured_output.invoke([system_message, human_message])
    
    new_iteration_count = state["iteration_count"] + 1
    
    if response.is_compliant:
        print("Auditor Status: APPROVED\n")
        return {
            "final_compliance_statement": state["current_draft"],
            "status": "APPROVED_BY_AI",
            "iteration_count": new_iteration_count,
            "feedback_history": [response]
        }
    else:
        print(f"Auditor Status: REJECTED. Issues found: {response.hallucinations_detected}\n")
        
        # Kill switch to prevent infinite loops
        if new_iteration_count >= 3:
            return {
                "status": "ESCALATED_TO_HUMAN",
                "iteration_count": new_iteration_count,
                "feedback_history": [response]
            }
            
        return {
            "status": "REJECTED",
            "iteration_count": new_iteration_count,
            "feedback_history": [response]
        }


# ---------------------------------------------------------
# 5. ROUTER (CONDITIONAL EDGE)
# ---------------------------------------------------------
def classifier(state: State):
    status = state.get("status")
    if status == "APPROVED_BY_AI":
        return END
    elif status == "ESCALATED_TO_HUMAN":
        return END
    else:
        return "writer"

# ---------------------------------------------------------
# 6. GRAPH COMPILATION
# ---------------------------------------------------------
graph_builder = StateGraph(State)
graph_builder.add_node("extractor", extractor)
graph_builder.add_node("retriever", retriever)
graph_builder.add_node("writer", writer)
graph_builder.add_node("auditor", auditor)

graph_builder.add_edge(START, "extractor")
graph_builder.add_edge("extractor", "retriever")
graph_builder.add_edge("retriever", "writer")
graph_builder.add_edge("writer", "auditor")
graph_builder.add_conditional_edges(
    "auditor", 
    classifier,
    {
        END: END,
        "writer": "writer"
    }
)

graph = graph_builder.compile()

# ---------------------------------------------------------
# 7. EXECUTION & TEST DATA
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Fake Test Data
    fake_client_email = "The structural glass must be able to withstand 180C heat without warping, and the architectural committee requires a 10% tint for aesthetic reasons."
    fake_database_spec = "Window-Model-X Performance Specs: Max temperature tolerance: 200C. Shading coefficient: 12% tint. UV Resistance: High."
    
    initial_state = {
        "requirements_text": fake_client_email,
        "product_sku": "Window-Model-X",
        "the_goal": "",
        "retrieved_datasheet_specs": fake_database_spec,
        "current_draft": "",
        "feedback_history": [],
        "iteration_count": 0,
        "final_compliance_statement": "",
        "status": "PENDING"
    }
    
    print("Starting Submittal Engine...\n" + "="*40)
    response = graph.invoke(initial_state)
    
    print("="*40)
    print(f"FINAL SYSTEM STATUS: {response['status']}")
    print(f"TOTAL ITERATIONS: {response['iteration_count']}")
    
    if response['status'] == "APPROVED_BY_AI":
        print(f"\nFINAL APPROVED DRAFT:\n{response['final_compliance_statement']}")
    elif response['status'] == "ESCALATED_TO_HUMAN":
        print(f"\nESCALATION REASON:\n{response['feedback_history'][-1].revision_instructions}")