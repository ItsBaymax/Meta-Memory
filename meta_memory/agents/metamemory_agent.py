from typing import Annotated, Literal, Sequence, TypedDict
import traceback
import sys, re

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_openai import ChatOpenAI


from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.tools import StructuredTool
# from langchain_core.pydantic_v1 import Field
from pydantic import BaseModel, Field
from tools.functions_wrapper import FunctionsWrapper


import sys, os
sys.path.append(sys.path[0] + '/..')


from utils.util import file_to_string
# from tools.tools import *

from memory.memory import Memory

from agents.agent import Agent, AgentOutput


### Print out state of the system
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    for k,v in state.items():
        if type(v) == str:
            print(v)

        elif type(v) == list:
            for item in v:
                if type(item) == str:
                    print(item)
                else:
                    print(item)
        else:
            print(item)

    # print(state)
    return state


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

def try_except_continue(state, func):
    while True:
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
            continue


class MetaMemoryAgent(Agent):

    def __init__(self, temperature=0, save_reasoning_path=None):
        self.temperature = temperature

        self.chat = FunctionsWrapper(ChatOpenAI(
                model="gpt-4o",
                temperature=temperature,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                # base_url="...",
                # organization="...",
                # other params...
            ))
        
        ### Load vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        # self.update_for_instance() # ref_time is None this time
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.agent_prompt = file_to_string(top_level_path+'prompts/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+'prompts/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(top_level_path+'prompts/agent_gen_system_prompt.txt')

        self.previous_tool_requests = "These are the tools you have previously used so far: \n"
        self.agent_call_count = 0

        self.chat_history = ChatMessageHistory()

        self.save_reasoning_path = save_reasoning_path

    def set_memory(self, memory: Memory):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()


    def create_tools(self, memory):
        class SemanticSimilarityRetrievalInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
Text embeddings of this description are used. There should always be text in here as a response! \
Based on the question and your context, decide what text to search for in the database. \
This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
The query will then search your memories for you.")

        self.semantic_similarity_retrieval_tool = StructuredTool.from_function(
            func=lambda x: memory.semantic_similarity_retrieval(x),
            name="semantic_similarity_retrieval",
            description="Search and return information from your memory in the form of captions",
            args_schema=SemanticSimilarityRetrievalInput
            # coroutine= ... <- you can specify an async method if desired as well
        )


        class SpatialRangeRetrievalInput(BaseModel):
            x: str = Field(description="The query will serve as the input for the spatial search module. \
You need to carefully consider which of the robot's previous position observations is most likely to reveal the object in question. After a thorough evaluation, provide your best guess. \
The search range is determined based on the assumption that the object in the question might exist within a certain radius around the chosen position. \
Example : If the question is, 'Where did I put my cup? but the context contains little to no information about the cup, while it does have information about the water dispenser, and the position of the water dispenser is (1, 2, 0). Therefore, the query should be: \
'question: Where did I put my cup?, reasoning: According to the robot's prior observations, there may be a cup within a 3-meter radius near position (1, 2, 0)., position: (1, 2, 0), search range: 3'. \
Note that the parameters question, position, and search range must be filled in the query.")    
        # position-based tool
        self.spatial_range_retrieval_tool = StructuredTool.from_function(
            func=lambda x: memory.spatial_range_retrieval(x),
            name="spatial_range_retrieval",
            description="Search and return information from your memory by using a position array such as (x,y,z)",
            args_schema=SpatialRangeRetrievalInput
            # coroutine= ... <- you can specify an async method if desired as well
        )


        class MemoryIntegrationInput(BaseModel):
            x: str = Field(description="The query will be used as input for the spatial awareness module. You must determine which type of input to use based on the question. There are four types in total: \
1. If the question is like this: 'Where is the nearest elevator?'. The start position is your current position, and the end position consists of the coordinates of all the elevators, and the candidate position is empty. \
For example, if your current position is (0.5, 0.2, 0.1), the input would be: 'question: Where is the nearest elevator?, start_to_end: [current position: (0.5, 0.2, 0.1), elevator_0: (0.4, 0.3, 0.1)], [current position: (0.5, 0.2, 0.1), elevator_1: (1.5, 1.2, 0.6)], [current position: (0.5, 0.2, 0.1), elevator_2: (2.1, 3.1, 1.6)], [...<more coordinate pairs>...], candidate position: []' \
2. If the question is like this: 'Where is the newspaper stand closest to the crosswalk?'. The start position should be the coordinates of the crosswalk, rather than your current position, and the end position should be the coordinates of all the newspaper stands. The candidate position is empty.\
For example: 'question: Where are the newspaper stands closest to the crosswalk?, start_to_end: [crosswalk_0: (1.5, 3.2, 2.1), newspaper stand_0: (0.7, 3.2, 1.1)], [crosswalk_1: (5.5, 4.2, 2.1), newspaper stand_1: (4.4, 2.8, 3.7)], [crosswalk_2: (7.5, 5.3, 2.1), newspaper stand_2: (3.8, 5.1, 3.7)], [...<more coordinate pairs>...], candidate position: []' \
3. If the question is like this: 'Where is the vending machine on my way to the basketball court?'. The start position is your current position, and the end position is the coordinate of the basketball court. The candidate position consists of the coordinates of all the vending machines. \
In this case, the coordinates for start position and end position can only have one pair. For example, if your current position is (0.5, 0.2, 0.1), the input would be: 'question: Where is the vending machine on my way to the basketball court?, start_to_end: [current position: (0.5, 0.2, 0.1), basketball court_0: (0.4, 0.3, 0.1)], candidate position: [vending machine_0: (0.5, 0.2, 0.1), vending machine_1: (1.5, 1.2, 0.6), vending machine_2: (0.8, 1.2, 2.1), <more candidate_position>]' \
4. If the question is like this: 'Where is the vending machine on the way from the soccer field to the basketball court?'. The start position should be the coordinate of the soccer field, rather than your current position, and the end position should be the coordinate of the basketball court. \
In this case, the coordinates for start position and end position can also only have one pair. For example: 'question: Where is the vending machine on the way from the soccer field to the basketball court?, start_to_end: [soccer field_0: (2.5, 1.2, 3.1), basketball court_0: (1.3, 1.7, 3.1)], candidate position: [vending machine_0: (0.5, 0.2, 0.1), vending machine_1: (1.5, 1.2, 0.6), vending machine_2: (1.3, 1.6, 0.8), <more candidate_position>]' \
5. If the question is like this: 'Where is the sidewalk in front of the red building?'. The start_to_end is empty, and the candidate position coordinates should include those of the sidewalks, red buildings.\
For example: 'question: Where is the sidewalk in front of the red building?, start_to_end: [], candidate position: [sidewalk_0: (0.5, 0.2, 0.1), red building_0: (1.5, 1.2, 0.6), sidewalk_1: (0.6, 2.5, 1.1), <more candidate_position>]\
You must ensure that the positions are relevant to the question, and on that basis, select the positions as comprehensively as possible.")

        self.memory_integration_tool = StructuredTool.from_function(
            func=lambda x: memory.memory_integration(x),
            name="memory_integration",
            description="Locate a coordinate based on the question and context.",
            args_schema=MemoryIntegrationInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        self.tool_list = [self.semantic_similarity_retrieval_tool, self.spatial_range_retrieval_tool, self.memory_integration_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]


    def agent(self, state):
        print("Agent Action.\n")
        with open(self.save_reasoning_path, 'a') as f:
            f.write("Agent Action.\n")
        messages = state["messages"]

        model = self.chat

        # limit to 5 tool calls.
        if self.agent_call_count < 3:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        elif 3 <= self.agent_call_count < 5:
            prompt = self.agent_gen_only_prompt
        else:
            raise SystemExit
        self.agent_call_count += 1

        agent_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )

        model = agent_prompt | model

        question = f"The question is: {messages[0]}"
        response = None
        try:
            response = model.invoke({"question": question, "chat_history": messages[:]})
        except Exception as e:
            print(f"Model response: {response}")
            raise e

        print(f'response: {response}\n')
        with open(self.save_reasoning_path, 'a') as f:
            f.write(f'response: {response}\n\n')

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    self.previous_tool_requests += f"You previously used the {tool_call['name']} tool with the arguments: {tool_call['args']['x']}. Therefore, you can no longer use these arguments as the query.\n"

        return {"messages": [response]}


    def generate(self, state):
        print("Generate Action.\n")
        with open(self.save_reasoning_path, 'a') as f:
            f.write("Generate Action.\n")
        messages = state["messages"]
        question = messages[0].content \
                + "\n Please responsed in the desired format."

        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question':question})

        gen_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        model = gen_prompt | self.chat

        response = model.invoke({"question": question, "chat_history": messages[1:]})

        print(f'response: {response}')
        with open(self.save_reasoning_path, 'a') as f:
            f.write(f'response: {response}\n\n')

        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = eval(response) 
            else:
                parsed = parse_json(response)

            # then check it has all the required keys
            keys_to_check_for = ["text", "position"]
            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")
            
            if type(parsed['position']) == str:
                parsed['position'] = eval(parsed['position'])
            
            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")

        except:
            raise ValueError("Generate call failed. Retrying...")

        self.previous_tool_requests = "These are the tools you have previously used so far: \n"
        self.agent_call_count = 0

        return {"messages": [str(parsed)]}


    def build_graph(self):

        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent

        # retrieve = ToolNode([self.retriever_tool])
        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)

        workflow.add_node("generate", lambda state: try_except_continue(state, self.generate))  
        # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not

        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "end": "generate",
            },
        )

        workflow.add_edge('action', 'agent')
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()


    def query(self, question: str):

        inputs = {"messages": [(("user", question)),]}

        out = self.graph.invoke(inputs)
        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = eval(response) 
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)

        return response


if __name__ == "__main__":

    from memory.milvus_memory import MilvusMemory

    memory = MilvusMemory("test", db_ip='127.0.0.1')
    agent = MetaMemoryAgent()
    agent.set_memory(memory)

    response = agent.query("Where can I sit?")
    response = agent.query_position("Where can I sit?")

