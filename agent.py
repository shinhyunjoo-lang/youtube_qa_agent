import os
import json
from typing import Optional
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

try:
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains.summarize import load_summarize_chain
    from langchain_classic.chains import RetrievalQA

# Global state for agent (will be set by YouTubeAgent instance)
_agent_state = {
    "llm": None,
    "docs": [],
    "vector_store": None,
    "qa_chain": None,
    "search": None,
    "video_context": "",
    "conversation_memory": {}
}

# ============================================================================
# TOOL DEFINITIONS using @tool decorator
# ============================================================================

@tool
def summarize_video(query: str = "") -> str:
    """Summarizes the entire YouTube video content. Use this when user asks for a general summary or overview."""
    if not _agent_state["docs"]:
        return "No video loaded."
    
    try:
        chain = load_summarize_chain(_agent_state["llm"], chain_type="map_reduce")
        return chain.run(_agent_state["docs"])
    except Exception as e:
        return f"Error summarizing: {str(e)}"

@tool
def generate_titles(query: str = "") -> str:
    """Generates catchy titles for the video. Use when user asks for title suggestions."""
    if not _agent_state["docs"]:
        return "No video loaded."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="Based on this video summary, generate 5 catchy titles:\n\n{summary}",
        input_variables=["summary"]
    )
    try:
        from langchain.chains import LLMChain
    except ImportError:
        from langchain_classic.chains import LLMChain
    chain = LLMChain(llm=_agent_state["llm"], prompt=prompt)
    return chain.run(summary)

@tool
def write_blog_post(query: str = "") -> str:
    """Writes a blog post based on the video content. Use when user asks for a blog post or article."""
    if not _agent_state["docs"]:
        return "No video loaded."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="Write a detailed blog post based on this video summary:\n\n{summary}",
        input_variables=["summary"]
    )
    try:
        from langchain.chains import LLMChain
    except ImportError:
        from langchain_classic.chains import LLMChain
    chain = LLMChain(llm=_agent_state["llm"], prompt=prompt)
    return chain.run(summary)

@tool
def generate_quiz(query: str = "") -> str:
    """Generates a multiple choice quiz based on video content. Use when user asks for a quiz or test."""
    if not _agent_state["docs"]:
        return "No video loaded."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="Create a 5-question multiple choice quiz based on this content:\n\n{summary}",
        input_variables=["summary"]
    )
    try:
        from langchain.chains import LLMChain
    except ImportError:
        from langchain_classic.chains import LLMChain
    chain = LLMChain(llm=_agent_state["llm"], prompt=prompt)
    return chain.run(summary)

@tool
def extract_key_moments(query: str = "") -> str:
    """Extracts key moments, topics, and takeaways from the video. Use when user asks for highlights or main points."""
    if not _agent_state["docs"]:
        return "No video loaded."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="Extract the key moments and main takeaways from this video:\n\n{summary}",
        input_variables=["summary"]
    )
    try:
        from langchain.chains import LLMChain
    except ImportError:
        from langchain_classic.chains import LLMChain
    chain = LLMChain(llm=_agent_state["llm"], prompt=prompt)
    return chain.run(summary)

@tool
def translate_content(target_language: str = "Korean") -> str:
    """Translates the video summary into the specified language. Default is Korean."""
    if not _agent_state["docs"]:
        return "No video loaded."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template=f"Translate the following text into {target_language}:\\n\\n{{text}}",
        input_variables=["text"]
    )
    try:
        from langchain.chains import LLMChain
    except ImportError:
        from langchain_classic.chains import LLMChain
    chain = LLMChain(llm=_agent_state["llm"], prompt=prompt)
    return chain.run(summary)

@tool
def search_web(query: str) -> str:
    """Searches the web for information NOT in the video. Use for current events, speaker background, or external facts. 
    IMPORTANT: Query must include specific entities/names from the video context."""
    if not _agent_state["search"]:
        return "Web search not available."
    
    try:
        return _agent_state["search"].invoke(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"

@tool
def store_memory(user_message: str) -> str:
    """Stores factual information provided by the user about the video (e.g., channel name, speaker name).
    ONLY use when user provides NEW facts as STATEMENTS, NOT for questions."""
    try:
        extraction_prompt = f"""Analyze the following user message and extract any factual information they are providing about the video.
Return ONLY a JSON object with key-value pairs. If no factual information is provided, return {{}}.

Examples:
- "이 비디오는 AWS Events 채널에 올라왔어" → {{"channel": "AWS Events"}}
- "발표자는 John이야" → {{"speaker": "John"}}
- "이건 re:Invent 2024 세션이야" → {{"event": "re:Invent 2024"}}

User message: {user_message}

Return JSON:"""
        
        response = _agent_state["llm"].invoke([HumanMessage(content=extraction_prompt)])
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        extracted_data = json.loads(content)
        
        if extracted_data:
            _agent_state["conversation_memory"].update(extracted_data)
            stored_keys = ", ".join(extracted_data.keys())
            return f"✓ 정보를 저장했습니다: {stored_keys}"
        else:
            return "저장할 새로운 정보가 없습니다."
            
    except Exception as e:
        return f"메모리 저장 중 오류: {str(e)}"

@tool
def answer_question(question: str) -> str:
    """Answers specific questions about the video content using the transcript. 
    Use for detailed questions about what was said in the video."""
    if not _agent_state["qa_chain"]:
        return "Vector store not initialized. Please reload video."
    
    try:
        return _agent_state["qa_chain"].run(question)
    except Exception as e:
        return f"Error answering question: {str(e)}"

# ============================================================================
# YOUTUBE AGENT CLASS
# ============================================================================

class YouTubeAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.docs = []
        self.vector_store = None
        self.qa_chain = None
        self.search = DuckDuckGoSearchRun()
        self.video_context = ""
        self.conversation_memory = {}
        self.video_id = ""
        
        # Update global state
        _agent_state["llm"] = self.llm
        _agent_state["search"] = self.search
        
        # Create agent with tools
        self.tools = [
            summarize_video,
            generate_titles,
            write_blog_post,
            generate_quiz,
            extract_key_moments,
            translate_content,
            search_web,
            store_memory,
            answer_question
        ]
        
        # Create ReAct agent
        self.agent = None
        self.agent_executor = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Sets up the ReAct agent with tools."""
        prompt_template = """You are a helpful YouTube video analysis assistant.

Current Video Context: {video_context}
User-Provided Information (Memory): {memory}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT RULES:
- If user is asking about stored information in Memory, answer directly without using tools
- For WebSearch, ALWAYS include specific entities/names from Video Context in the query
- For store_memory, ONLY use when user provides STATEMENTS with new facts, NOT for questions
- If user message contains ?, 뭐, 무엇, 어디, 누구, DO NOT use store_memory

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad", "video_context", "memory", "tools", "tool_names"]
        )
        
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def load_video(self, url):
        """Loads the video transcript."""
        try:
            self.docs = []
            self.vector_store = None
            self.qa_chain = None
            self.video_context = ""

            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["en", "en-US", "ko"])
            self.docs = loader.load()
            
            if not self.docs:
                return "Error: No transcript found for this video. Please check if the video has English or Korean captions."
                
            title = self.docs[0].metadata.get('title', 'YouTube Video')
            self.docs[0].metadata['title'] = title
            self.video_context = f"Video Title: {title}"
            
            # Extract video ID
            if "v=" in url:
                self.video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                self.video_id = url.split("youtu.be/")[1].split("?")[0]
            else:
                import hashlib
                self.video_id = hashlib.md5(url.encode()).hexdigest()[:12]
            
            # Update global state
            _agent_state["docs"] = self.docs
            _agent_state["video_context"] = self.video_context
                
            return f"Successfully loaded video: {title}"
        except Exception as e:
            return f"Error loading video: {str(e)}"

    def create_vector_store(self):
        """Creates a FAISS vector store, using local persistence."""
        if not self.docs:
            return "No documents to process. Load a video first."
        
        db_path = f"db/{self.video_id}"
        
        if os.path.exists(db_path):
            try:
                embeddings = OpenAIEmbeddings()
                self.vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
                print(f"DEBUG: Loaded existing vector store from {db_path}")
            except Exception as e:
                print(f"Warning: Could not load local index: {e}. Recreating.")
        
        if not self.vector_store:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(self.docs)
            
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(split_docs, embeddings)
            self.vector_store.save_local(db_path)
            print(f"DEBUG: Saved new vector store to {db_path}")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        
        # Generate context summary
        if "Summary:" not in self.video_context:
            try:
                brief_content = self.docs[0].page_content[:3000]
                summary_prompt = f"In 1-2 sentences, what is this video about?\n\n{brief_content}"
                context_summary = self.llm.invoke([HumanMessage(content=summary_prompt)]).content
                self.video_context += f"\nSummary: {context_summary}"
                print(f"DEBUG: Generated context summary")
            except Exception as e:
                print(f"Warning: Could not generate context summary: {e}")
        
        # Update global state
        _agent_state["vector_store"] = self.vector_store
        _agent_state["qa_chain"] = self.qa_chain
        _agent_state["video_context"] = self.video_context
        
        return "Vector store created/loaded successfully."

    def run(self, query):
        """Entry point for the agent using AgentExecutor."""
        if not self.docs:
            return "Please load a video first."
        
        # Pre-check: answer from memory if applicable
        if self.conversation_memory and any(keyword in query.lower() for keyword in ['뭐', '무엇', '어떤', '누구', 'what', 'who', 'which']):
            try:
                memory_check_prompt = f"""User has stored the following information:
{json.dumps(self.conversation_memory, ensure_ascii=False, indent=2)}

User question: {query}

If the question is asking about information that exists in the stored data above, answer it directly and concisely in Korean.
If the information is NOT in the stored data, respond with "NOT_FOUND".

Answer:"""
                
                response = self.llm.invoke([HumanMessage(content=memory_check_prompt)])
                answer = response.content.strip()
                
                if answer != "NOT_FOUND" and "NOT_FOUND" not in answer:
                    return f"💾 저장된 정보: {answer}"
            except Exception as e:
                print(f"Memory check error: {e}")
        
        # Update global state before running agent
        _agent_state["conversation_memory"] = self.conversation_memory
        
        # Run agent
        try:
            memory_str = json.dumps(self.conversation_memory, ensure_ascii=False, indent=2) if self.conversation_memory else "None"
            result = self.agent_executor.invoke({
                "input": query,
                "video_context": self.video_context,
                "memory": memory_str
            })
            return result["output"]
        except Exception as e:
            return f"Agent Error: {str(e)}"
