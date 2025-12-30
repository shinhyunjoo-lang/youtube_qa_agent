import os
import json
from typing import Optional
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
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
        return "비디오가 로드되지 않았습니다."
    
    try:
        # Create a Korean-specific summarization prompt
        korean_prompt = PromptTemplate(
            template="다음 비디오 내용을 한국어로 요약해주세요. 주요 내용과 핵심 포인트를 포함해서 자세히 설명해주세요:\n\n{text}",
            input_variables=["text"]
        )
        
        chain = load_summarize_chain(_agent_state["llm"], chain_type="map_reduce", map_prompt=korean_prompt, combine_prompt=korean_prompt)
        return chain.run(_agent_state["docs"])
    except Exception as e:
        return f"요약 중 오류가 발생했습니다: {str(e)}"

@tool
def generate_titles(query: str = "") -> str:
    """Generates catchy titles for the video. Use when user asks for title suggestions."""
    if not _agent_state["docs"]:
        return "비디오가 로드되지 않았습니다."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="다음 비디오 요약을 바탕으로 매력적인 한국어 제목 5개를 생성해주세요:\n\n{summary}",
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
        return "비디오가 로드되지 않았습니다."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="다음 비디오 요약을 바탕으로 상세한 한국어 블로그 포스트를 작성해주세요. 서론, 본론, 결론 구조로 작성해주세요:\n\n{summary}",
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
        return "비디오가 로드되지 않았습니다."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="다음 내용을 바탕으로 한국어로 5문항의 객관식 퀴즈를 만들어주세요. 각 문항마다 4개의 선택지와 정답을 포함해주세요:\n\n{summary}",
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
        return "비디오가 로드되지 않았습니다."
    
    summary = summarize_video.invoke("")
    prompt = PromptTemplate(
        template="다음 비디오에서 핵심 순간들과 주요 내용을 한국어로 추출해주세요. 중요한 포인트들을 시간순으로 정리해주세요:\n\n{summary}",
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
        return "웹 검색을 사용할 수 없습니다."
    
    try:
        # Get video context for better search
        video_context = _agent_state.get("video_context", "")
        conversation_memory = _agent_state.get("conversation_memory", {})
        docs = _agent_state.get("docs", [])
        
        # Extract key terms from video content for better search context
        search_context_terms = []
        
        # Get video title
        if "Video Title: " in video_context:
            title = video_context.split("Video Title: ")[1].split("\n")[0]
            if title and title != "YouTube Video":
                search_context_terms.append(f'"{title}"')
        
        # Add memory context
        if conversation_memory:
            for key, value in conversation_memory.items():
                if key == "channel":
                    search_context_terms.append(f'"{value}"')
                elif key == "speaker":
                    search_context_terms.append(f'"{value}"')
                elif key == "event":
                    search_context_terms.append(f'"{value}"')
        
        # Extract key terms from video content (first 1000 characters)
        if docs and len(docs) > 0:
            content_sample = docs[0].page_content[:1000]
            # Use LLM to extract key terms for search
            key_terms_prompt = f"""다음 비디오 내용에서 검색에 유용한 핵심 키워드 3-5개를 추출해주세요. 
회사명, 제품명, 기술명, 인물명 등을 우선적으로 추출하세요.
키워드만 쉼표로 구분해서 답변해주세요.

비디오 내용:
{content_sample}

키워드:"""
            
            key_terms_response = _agent_state["llm"].invoke([HumanMessage(content=key_terms_prompt)])
            key_terms = key_terms_response.content.strip()
            if key_terms and len(key_terms) > 5:
                search_context_terms.append(key_terms)
        
        # Create enhanced search query
        if search_context_terms:
            enhanced_query = f"{query} {' '.join(search_context_terms)}"
        else:
            enhanced_query = query
            
        print(f"DEBUG: Original query: {query}")
        print(f"DEBUG: Enhanced search query: {enhanced_query}")
        
        search_result = _agent_state["search"].invoke(enhanced_query)
        
        # Filter and contextualize search result
        filter_prompt = f"""다음 검색 결과를 분석하여 현재 비디오와 관련된 내용만 선별하고 한국어로 번역/요약해주세요.

현재 비디오 정보:
- 제목: {video_context}
- 사용자 제공 정보: {json.dumps(conversation_memory, ensure_ascii=False)}
- 비디오 주요 내용: {content_sample[:500] if docs and len(docs) > 0 else '정보 없음'}

검색 결과:
{search_result}

관련성이 높은 내용만 선별하여 한국어로 요약해주세요. 관련성이 낮은 내용은 제외하고, 
현재 비디오와 직접적으로 연관된 기사나 정보만 포함해주세요:"""
        
        response = _agent_state["llm"].invoke([HumanMessage(content=filter_prompt)])
        return response.content
        
    except Exception as e:
        return f"웹 검색 실패: {str(e)}"

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
        return "벡터 스토어가 초기화되지 않았습니다. 비디오를 다시 로드해주세요."
    
    try:
        # Create a Korean-specific QA prompt
        korean_qa_prompt = PromptTemplate(
            template="""다음 컨텍스트를 바탕으로 질문에 한국어로 답변해주세요. 답변은 정확하고 상세하게 해주세요.

컨텍스트: {context}

질문: {question}

답변:""",
            input_variables=["context", "question"]
        )
        
        # Update QA chain with Korean prompt
        try:
            from langchain.chains import RetrievalQA
        except ImportError:
            from langchain_classic.chains import RetrievalQA
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=_agent_state["llm"],
            chain_type="stuff",
            retriever=_agent_state["vector_store"].as_retriever(),
            chain_type_kwargs={"prompt": korean_qa_prompt}
        )
        
        return qa_chain.run(question)
    except Exception as e:
        return f"질문 답변 중 오류가 발생했습니다: {str(e)}"

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
        
        # Create tools mapping for simple routing
        self.tools = {
            "summarize": summarize_video,
            "titles": generate_titles,
            "blog": write_blog_post,
            "quiz": generate_quiz,
            "moments": extract_key_moments,
            "translate": translate_content,
            "search": search_web,
            "memory": store_memory,
            "answer": answer_question
        }
    
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
                return "오류: 이 비디오의 자막을 찾을 수 없습니다. 영어 또는 한국어 자막이 있는지 확인해주세요."
                
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
                
            return f"비디오를 성공적으로 로드했습니다: {title}"
        except Exception as e:
            return f"비디오 로드 중 오류가 발생했습니다: {str(e)}"

    def create_vector_store(self):
        """Creates a FAISS vector store, using local persistence."""
        if not self.docs:
            return "처리할 문서가 없습니다. 먼저 비디오를 로드해주세요."
        
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
        
        # Create Korean QA chain
        korean_qa_prompt = PromptTemplate(
            template="""다음 컨텍스트를 바탕으로 질문에 한국어로 답변해주세요. 답변은 정확하고 상세하게 해주세요.

컨텍스트: {context}

질문: {question}

답변:""",
            input_variables=["context", "question"]
        )
        
        try:
            from langchain.chains import RetrievalQA
        except ImportError:
            from langchain_classic.chains import RetrievalQA
            
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": korean_qa_prompt}
        )
        
        # Generate context summary in Korean
        if "Summary:" not in self.video_context:
            try:
                brief_content = self.docs[0].page_content[:3000]
                summary_prompt = f"다음 비디오 내용을 1-2문장으로 한국어로 요약해주세요:\n\n{brief_content}"
                context_summary = self.llm.invoke([HumanMessage(content=summary_prompt)]).content
                self.video_context += f"\nSummary: {context_summary}"
                print(f"DEBUG: Generated context summary")
            except Exception as e:
                print(f"Warning: Could not generate context summary: {e}")
        
        # Update global state
        _agent_state["vector_store"] = self.vector_store
        _agent_state["qa_chain"] = self.qa_chain
        _agent_state["video_context"] = self.video_context
        
        return "벡터 스토어가 성공적으로 생성/로드되었습니다."

    def run(self, query):
        """Simple routing-based agent without complex agent framework."""
        if not self.docs:
            return "먼저 비디오를 로드해주세요."
        
        # Pre-check: answer from memory if applicable
        if self.conversation_memory and any(keyword in query.lower() for keyword in ['뭐', '무엇', '어떤', '누구', '어디', '언제', 'what', 'who', 'which', 'where', 'when', '채널']):
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
        
        # Update global state before processing
        _agent_state["conversation_memory"] = self.conversation_memory
        
        # Check if user is providing information (statements, not questions)
        if not any(q_word in query for q_word in ['?', '뭐', '무엇', '어디', '누구', '언제', '어떻게', 'what', 'who', 'where', 'when', 'how']):
            # This might be a statement providing information
            try:
                result = store_memory.invoke(query)
                if "정보를 저장했습니다" in result:
                    # Update local memory as well
                    extraction_prompt = f"""Analyze the following user message and extract any factual information they are providing about the video.
Return ONLY a JSON object with key-value pairs. If no factual information is provided, return {{}}.

Examples:
- "이 비디오는 AWS Events 채널에 올라왔어" → {{"channel": "AWS Events"}}
- "발표자는 John이야" → {{"speaker": "John"}}
- "이건 re:Invent 2024 세션이야" → {{"event": "re:Invent 2024"}}

User message: {query}

Return JSON:"""
                    
                    response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
                    content = response.content.strip()
                    
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].strip()
                    
                    try:
                        extracted_data = json.loads(content)
                        if extracted_data:
                            self.conversation_memory.update(extracted_data)
                    except:
                        pass
                    
                    return result
            except Exception as e:
                print(f"Memory storage error: {e}")
        
        # Simple routing logic
        query_lower = query.lower()
        
        # Check for specific tool requests
        if any(word in query_lower for word in ['요약', 'summary', 'summarize']):
            return summarize_video.invoke("")
        elif any(word in query_lower for word in ['제목', 'title', 'titles']):
            return generate_titles.invoke("")
        elif any(word in query_lower for word in ['블로그', 'blog', 'post']):
            return write_blog_post.invoke("")
        elif any(word in query_lower for word in ['퀴즈', 'quiz', 'test']):
            return generate_quiz.invoke("")
        elif any(word in query_lower for word in ['핵심', '중요', 'key', 'moments', 'highlights']):
            return extract_key_moments.invoke("")
        elif any(word in query_lower for word in ['번역', 'translate', '한국어']):
            return translate_content.invoke("Korean")
        elif any(word in query_lower for word in ['검색', 'search', '찾아', '기사', '뉴스', '관련', '출처']):
            # Don't modify the original query, just use it as is for search
            print(f"DEBUG: Search triggered by query: {query}")
            
            # If the query is asking for articles or sources, create a better search query
            if any(word in query_lower for word in ['기사', '뉴스', '관련', '출처']):
                # Use the original query for search
                search_query = query
            else:
                # Extract search terms from query and enhance with video context
                search_query = query.replace('검색', '').replace('찾아', '').replace('search', '').strip()
                
                # If no specific search terms after cleaning, use video context
                if not search_query or len(search_query) < 3:
                    if self.conversation_memory:
                        # Use stored information for search
                        search_terms = []
                        for key, value in self.conversation_memory.items():
                            search_terms.append(str(value))
                        search_query = " ".join(search_terms) + " 관련 기사"
                    else:
                        # Use video title
                        title = self.docs[0].metadata.get('title', '') if self.docs else ''
                        search_query = f"{title} 관련 기사" if title else "검색할 내용을 구체적으로 알려주세요."
            
            if search_query and search_query != "검색할 내용을 구체적으로 알려주세요.":
                return search_web.invoke(search_query)
            else:
                return "검색할 내용을 구체적으로 알려주세요."
        else:
            # Default to answering questions about the video
            return answer_question.invoke(query)
