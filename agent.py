import os
import json
from typing import Optional
from datetime import datetime
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document

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
    "conversation_memory": {},
    "video_info": {}
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
        print(f"📝 [TOOL] 비디오 요약 생성 중...")
        
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
    
    print(f"🎯 [TOOL] 제목 생성 중...")
    
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
    
    print(f"✍️ [TOOL] 블로그 포스트 작성 중...")
    
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
    
    print(f"❓ [TOOL] 퀴즈 생성 중...")
    
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
    
    print(f"⭐ [TOOL] 핵심 순간 추출 중...")
    
    # Use the full transcript instead of summary for better timestamp extraction
    full_content = _agent_state["docs"][0].page_content if _agent_state["docs"] else ""
    
    prompt = PromptTemplate(
        template="""다음 비디오 전체 내용에서 핵심 순간들과 주요 내용을 한국어로 추출해주세요. 
각 핵심 순간마다 타임스탬프를 "0:30", "1:45", "3:20" 형식으로 포함하여 시간순으로 정리해주세요.

비디오 내용:
{content}

다음 형식으로 답변해주세요:
**0:00 - 시작 부분**
- 주요 내용 설명

**1:30 - 중간 부분**  
- 주요 내용 설명

**3:45 - 마무리 부분**
- 주요 내용 설명

핵심 순간들:""",
        input_variables=["content"]
    )
    try:
        from langchain.chains import LLMChain
    except ImportError:
        from langchain_classic.chains import LLMChain
    chain = LLMChain(llm=_agent_state["llm"], prompt=prompt)
    return chain.run(full_content[:4000])  # Limit content length for better processing

@tool
def search_web(query: str) -> str:
    """Searches the web for information NOT in the video. Use for current events, speaker background, or external facts."""
    if not _agent_state["search"]:
        return "웹 검색을 사용할 수 없습니다."
    
    try:
        print(f"🔍 [TOOL] 웹 검색 실행 중...")
        
        # Get comprehensive video context
        video_context = _agent_state.get("video_context", "")
        conversation_memory = _agent_state.get("conversation_memory", {})
        video_info = _agent_state.get("video_info", {})
        docs = _agent_state.get("docs", [])
        
        # Extract detailed context from video content
        search_context = []
        
        # 1. Get video title and author from video_info
        video_title = video_info.get("title", "")
        video_author = video_info.get("author", "")
        
        if video_title and video_title != "YouTube Video":
            search_context.append(video_title)
        if video_author:
            search_context.append(video_author)
        
        # 2. Add stored memory context (channel, speaker, event info)
        memory_context = []
        if conversation_memory:
            for key, value in conversation_memory.items():
                memory_context.append(f"{key}: {value}")
                search_context.append(str(value))
        
        # 3. Extract key entities from video content using LLM
        video_entities = []
        if docs and len(docs) > 0:
            content_sample = docs[0].page_content[:2000]  # Use more content for better context
            
            entity_extraction_prompt = f"""다음 비디오 내용에서 웹 검색에 유용한 핵심 엔티티들을 추출해주세요.
다음 카테고리별로 추출하세요:
- 회사명/조직명
- 제품명/서비스명  
- 기술명/플랫폼명
- 인물명
- 이벤트명/컨퍼런스명
- 주요 키워드

비디오 내용:
{content_sample}

각 카테고리별로 찾은 엔티티들을 쉼표로 구분하여 나열해주세요. 없으면 "없음"이라고 하세요.

회사명/조직명:
제품명/서비스명:
기술명/플랫폼명:
인물명:
이벤트명/컨퍼런스명:
주요 키워드:"""
            
            try:
                entity_response = _agent_state["llm"].invoke([HumanMessage(content=entity_extraction_prompt)])
                entity_text = entity_response.content.strip()
                
                # Parse extracted entities
                for line in entity_text.split('\n'):
                    if ':' in line and '없음' not in line.lower():
                        entities = line.split(':', 1)[1].strip()
                        if entities and len(entities) > 3:
                            video_entities.extend([e.strip() for e in entities.split(',') if e.strip()])
                
                # Add top entities to search context
                search_context.extend(video_entities[:5])  # Top 5 entities
                
            except Exception as e:
                print(f"Entity extraction error: {e}")
        
        # 4. Create intelligent search query
        if not search_context:
            return "🚫 검색할 구체적인 정보가 부족합니다. 비디오 내용을 더 구체적으로 분석한 후 다시 시도해주세요."
        
        # Build search query with context
        context_terms = " ".join(search_context[:4])  # Use top 4 context terms
        
        # Smart query construction based on user intent
        if any(word in query.lower() for word in ['관련', '더', '자세한', '설명', '정보']):
            # User wants more detailed information about the video topic
            final_query = f"{context_terms} 자세한 설명 최신 정보"
        elif any(word in query.lower() for word in ['기사', '뉴스', '소식']):
            # User wants news/articles
            final_query = f"{context_terms} 뉴스 기사 최신"
        elif any(word in query.lower() for word in ['배경', '역사', '소개']):
            # User wants background information
            final_query = f"{context_terms} 배경 소개 개요"
        else:
            # General search with context
            final_query = f"{query} {context_terms}"
            
        print(f"🔍 검색어: {final_query}")
        print(f"📋 추출된 컨텍스트: {', '.join(search_context[:5])}")
        
        # Perform web search
        search_result = _agent_state["search"].invoke(final_query)
        
        if not search_result or len(search_result.strip()) < 50:
            return "🚫 관련 검색 결과를 찾을 수 없습니다. 다른 키워드로 시도해보세요."
        
        # Enhanced result filtering with video context
        detailed_info_available = video_info.get('detailed_info_available', False)
        
        filter_prompt = f"""다음 웹 검색 결과를 분석하여 현재 로드된 비디오와 관련된 정보만 선별해주세요.

현재 비디오 정보:
- 제목: {video_title}"""
        
        if detailed_info_available:
            filter_prompt += f"""
- 채널/작성자: {video_author}
- 조회수: {video_info.get('view_count', '정보 없음')}
- 길이: {video_info.get('length', '정보 없음')}
- 게시일: {video_info.get('publish_date', '정보 없음')}
- 설명: {video_info.get('description', '')[:200]}..."""
        else:
            filter_prompt += "\n- 상세 정보: 기본 정보만 사용 가능"
            
        filter_prompt += f"""
- 저장된 정보: {json.dumps(conversation_memory, ensure_ascii=False)}
- 추출된 주요 엔티티: {', '.join(video_entities[:10])}
- 사용자 질문: {query}

검색 결과:
{search_result}

다음 기준으로 정보를 정리해주세요:
1. 위 비디오 정보와 직접 관련된 내용만 선별
2. 비디오에서 언급된 회사, 제품, 기술, 인물, 이벤트와 관련된 정보 우선
3. 최신 정보 및 공식 발표 내용 포함
4. 한국어로 명확하고 구체적으로 요약
5. 관련 없는 일반적인 정보나 광고는 제외
6. 출처나 날짜가 있다면 포함

비디오와 관련된 구체적인 정보:"""
        
        response = _agent_state["llm"].invoke([HumanMessage(content=filter_prompt)])
        result = response.content.strip()
        
        if len(result) < 50 or "관련된 정보를 찾을 수 없" in result:
            return f"🚫 '{video_title}'과 관련된 구체적인 웹 정보를 찾을 수 없습니다. 다른 검색어로 시도해보세요."
            
        return f"🌐 웹 검색 결과 ('{video_title}' 관련):\n\n{result}"
        
    except Exception as e:
        print(f"❌ 웹 검색 오류: {str(e)}")
        return f"웹 검색 중 오류가 발생했습니다: {str(e)}"

@tool
def store_memory(user_message: str) -> str:
    """Stores factual information provided by the user about the video (e.g., channel name, speaker name).
    ONLY use when user provides NEW facts as STATEMENTS, NOT for questions."""
    try:
        print(f"💾 [TOOL] 메모리 저장 중...")
        
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
            # 1. 메모리에 저장 (기존 방식)
            _agent_state["conversation_memory"].update(extracted_data)
            
            # 2. 벡터 스토어에도 검색 가능한 형태로 추가
            if _agent_state["vector_store"]:
                memory_docs = []
                for key, value in extracted_data.items():
                    # 자연어 형태로 변환하여 벡터화
                    if key == "channel":
                        doc_content = f"이 비디오는 {value} 채널에서 제공됩니다."
                    elif key == "speaker":
                        doc_content = f"이 비디오의 발표자는 {value}입니다."
                    elif key == "event":
                        doc_content = f"이 비디오는 {value} 이벤트의 일부입니다."
                    else:
                        doc_content = f"이 비디오의 {key}는 {value}입니다."
                    
                    memory_doc = Document(
                        page_content=doc_content,
                        metadata={"type": "memory", "key": key, "value": value}
                    )
                    memory_docs.append(memory_doc)
                
                # 벡터 스토어에 추가
                _agent_state["vector_store"].add_documents(memory_docs)
                print(f"DEBUG: Added {len(memory_docs)} memory documents to vector store")
            
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
        print(f"🤔 [TOOL] 비디오 내용 질문 답변 중...")
        
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
        self.llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
        self.docs = []
        self.vector_store = None
        self.qa_chain = None
        self.search = DuckDuckGoSearchRun()
        self.video_context = ""
        self.conversation_memory = {}
        self.video_info = {}
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
            "search": search_web,
            "memory": store_memory,
            "answer": answer_question
        }
    
    def save_metadata(self):
        """Save conversation memory, context, and video info to JSON file."""
        try:
            db_path = f"db/{self.video_id}"
            os.makedirs(db_path, exist_ok=True)
            
            # Extract video info from docs metadata if available
            video_info = {}
            if self.docs and len(self.docs) > 0:
                doc_metadata = self.docs[0].metadata
                video_info = {
                    "title": doc_metadata.get("title", ""),
                    "description": doc_metadata.get("description", ""),
                    "view_count": doc_metadata.get("view_count", ""),
                    "length": doc_metadata.get("length", ""),
                    "author": doc_metadata.get("author", ""),
                    "publish_date": doc_metadata.get("publish_date", ""),
                    "upload_date": doc_metadata.get("upload_date", "")
                }
            
            metadata = {
                "conversation_memory": self.conversation_memory,
                "video_context": self.video_context,
                "video_info": video_info,
                "timestamp": datetime.now().isoformat(),
                "video_id": self.video_id
            }
            
            metadata_path = f"{db_path}/metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"DEBUG: Saved metadata with video info to {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def load_metadata(self):
        """Load conversation memory, context, and video info from JSON file."""
        try:
            metadata_path = f"db/{self.video_id}/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                self.conversation_memory = metadata.get("conversation_memory", {})
                saved_context = metadata.get("video_context", "")
                self.video_info = metadata.get("video_info", {})
                
                # 기존 컨텍스트와 병합
                if saved_context and saved_context != self.video_context:
                    self.video_context = saved_context
                
                print(f"DEBUG: Loaded metadata from {metadata_path}")
                print(f"DEBUG: Loaded conversation memory: {self.conversation_memory}")
                print(f"DEBUG: Loaded video info: {self.video_info}")
                
                # 글로벌 상태 업데이트
                _agent_state["conversation_memory"] = self.conversation_memory
                _agent_state["video_context"] = self.video_context
                _agent_state["video_info"] = getattr(self, 'video_info', {})
                
                return True
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
        
        return False
    
    def add_video_info_to_vector_store(self, video_info):
        """Add video information to vector store for semantic search."""
        if self.vector_store and video_info:
            try:
                video_docs = []
                
                # Add title
                if video_info.get("title"):
                    title_doc = Document(
                        page_content=f"비디오 제목: {video_info['title']}",
                        metadata={"type": "video_info", "info_type": "title"}
                    )
                    video_docs.append(title_doc)
                
                # Add description
                if video_info.get("description"):
                    desc_doc = Document(
                        page_content=f"비디오 설명: {video_info['description'][:500]}",  # Limit description length
                        metadata={"type": "video_info", "info_type": "description"}
                    )
                    video_docs.append(desc_doc)
                
                # Add author/channel info
                if video_info.get("author"):
                    author_doc = Document(
                        page_content=f"비디오 채널/작성자: {video_info['author']}",
                        metadata={"type": "video_info", "info_type": "author"}
                    )
                    video_docs.append(author_doc)
                
                # Add view count and length info
                stats_info = []
                if video_info.get("view_count"):
                    stats_info.append(f"조회수: {video_info['view_count']}")
                if video_info.get("length"):
                    stats_info.append(f"길이: {video_info['length']}")
                if video_info.get("publish_date"):
                    stats_info.append(f"게시일: {video_info['publish_date']}")
                
                if stats_info:
                    stats_doc = Document(
                        page_content=f"비디오 정보: {', '.join(stats_info)}",
                        metadata={"type": "video_info", "info_type": "stats"}
                    )
                    video_docs.append(stats_doc)
                
                if video_docs:
                    self.vector_store.add_documents(video_docs)
                    print(f"DEBUG: Added {len(video_docs)} video info documents to vector store")
                    
            except Exception as e:
                print(f"Warning: Could not add video info to vector store: {e}")
    
    def load_video(self, url):
        """Loads the video transcript with detailed video information."""
        try:
            self.docs = []
            self.vector_store = None
            self.qa_chain = None
            self.video_context = ""
            self.video_info = {}

            # First try with detailed info, fallback to basic if it fails
            try:
                print("🔄 비디오 상세 정보와 함께 로드 시도 중...")
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=["en", "en-US", "ko"])
                self.docs = loader.load()
                detailed_info_loaded = True
                print("✅ 상세 정보 로드 성공")
            except Exception as e:
                print(f"⚠️ 상세 정보 로드 실패: {e}")
                print("🔄 기본 정보로 재시도 중...")
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["en", "en-US", "ko"])
                self.docs = loader.load()
                detailed_info_loaded = False
                print("✅ 기본 정보 로드 성공")
            
            if not self.docs:
                return "오류: 이 비디오의 자막을 찾을 수 없습니다. 영어 또는 한국어 자막이 있는지 확인해주세요."
            
            # Extract video information from metadata
            doc_metadata = self.docs[0].metadata
            print(f"DEBUG: Available metadata keys: {list(doc_metadata.keys())}")
            print(f"DEBUG: Metadata content: {doc_metadata}")
            
            # Extract basic info that should always be available
            title = doc_metadata.get('title', 'YouTube Video')
            source = doc_metadata.get('source', '')
            
            # Extract detailed info if available
            description = doc_metadata.get('description', '') if detailed_info_loaded else ''
            author = doc_metadata.get('author', '') if detailed_info_loaded else ''
            view_count = doc_metadata.get('view_count', '') if detailed_info_loaded else ''
            length = doc_metadata.get('length', '') if detailed_info_loaded else ''
            publish_date = doc_metadata.get('publish_date', '') if detailed_info_loaded else ''
            upload_date = doc_metadata.get('upload_date', '') if detailed_info_loaded else ''
            
            # If basic info failed, try to extract from source URL or other fields
            if title == 'YouTube Video' or not title:
                # Try to get title from other metadata fields
                for key in ['video_title', 'name', 'display_name']:
                    if key in doc_metadata and doc_metadata[key]:
                        title = doc_metadata[key]
                        break
                
                # If still no title, extract video ID from URL for identification
                if title == 'YouTube Video' or not title:
                    if "v=" in url:
                        video_id_from_url = url.split("v=")[1].split("&")[0]
                        title = f"YouTube Video ({video_id_from_url})"
                    elif "youtu.be/" in url:
                        video_id_from_url = url.split("youtu.be/")[1].split("?")[0]
                        title = f"YouTube Video ({video_id_from_url})"
            
            # Store video info
            self.video_info = {
                "title": title,
                "description": description,
                "author": author,
                "view_count": view_count,
                "length": length,
                "publish_date": publish_date,
                "upload_date": upload_date,
                "source": source,
                "detailed_info_available": detailed_info_loaded
            }
            
            print(f"DEBUG: Extracted video info: {self.video_info}")
            
            # Create video context based on available information
            context_parts = [f"Video Title: {title}"]
            if author:
                context_parts.append(f"Channel: {author}")
            if view_count:
                context_parts.append(f"Views: {view_count}")
            if length:
                context_parts.append(f"Length: {length}")
            if publish_date:
                context_parts.append(f"Published: {publish_date}")
            if description and len(description) > 0:
                # Add first 200 characters of description
                desc_preview = description[:200] + "..." if len(description) > 200 else description
                context_parts.append(f"Description: {desc_preview}")
            
            self.video_context = "\n".join(context_parts)
            
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
            _agent_state["video_info"] = self.video_info
            
            # Display loaded video info
            info_display = [f"📺 비디오를 성공적으로 로드했습니다: {title}"]
            
            if detailed_info_loaded:
                info_display.append("✅ 상세 정보 포함")
                if author:
                    info_display.append(f"📺 채널: {author}")
                if view_count:
                    info_display.append(f"👀 조회수: {view_count}")
                if length:
                    info_display.append(f"⏱️ 길이: {length}")
                if publish_date:
                    info_display.append(f"📅 게시일: {publish_date}")
            else:
                info_display.append("⚠️ 기본 정보만 로드됨 (상세 정보 로드 실패)")
                
            return "\n".join(info_display)
            
        except Exception as e:
            return f"비디오 로드 중 오류가 발생했습니다: {str(e)}"

    def create_vector_store(self):
        """Creates a FAISS vector store, using local persistence."""
        if not self.docs:
            return "처리할 문서가 없습니다. 먼저 비디오를 로드해주세요."
        
        db_path = f"db/{self.video_id}"
        vector_store_loaded = False
        
        # 1. 메타데이터 로드 (JSON)
        metadata_loaded = self.load_metadata()
        
        if os.path.exists(db_path):
            try:
                embeddings = OpenAIEmbeddings()
                self.vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
                print(f"DEBUG: Loaded existing vector store from {db_path}")
                vector_store_loaded = True
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
        
        # 2. Add video information to vector store (for new vector stores)
        if not vector_store_loaded and hasattr(self, 'video_info') and self.video_info:
            # Only add detailed info if it was successfully loaded
            if self.video_info.get('detailed_info_available', False):
                self.add_video_info_to_vector_store(self.video_info)
            else:
                # Add basic title info to vector store
                if self.video_info.get('title'):
                    basic_info_doc = Document(
                        page_content=f"비디오 제목: {self.video_info['title']}",
                        metadata={"type": "video_info", "info_type": "title"}
                    )
                    self.vector_store.add_documents([basic_info_doc])
                    print("DEBUG: Added basic video title to vector store")
        
        # 3. Generate and add context summary to vector store
        if "Summary:" not in self.video_context:
            try:
                brief_content = self.docs[0].page_content[:3000]
                summary_prompt = f"다음 비디오 내용을 1-2문장으로 한국어로 요약해주세요:\n\n{brief_content}"
                context_summary = self.llm.invoke([HumanMessage(content=summary_prompt)]).content
                self.video_context += f"\nSummary: {context_summary}"
                
                # Add context summary to vector store for semantic search
                self.add_context_to_vector_store(context_summary)
                print(f"DEBUG: Generated context summary")
            except Exception as e:
                print(f"Warning: Could not generate context summary: {e}")
        
        # 4. Add existing conversation memory to vector store (if loaded from JSON)
        if metadata_loaded and self.conversation_memory:
            try:
                memory_docs = []
                for key, value in self.conversation_memory.items():
                    if key == "channel":
                        doc_content = f"이 비디오는 {value} 채널에서 제공됩니다."
                    elif key == "speaker":
                        doc_content = f"이 비디오의 발표자는 {value}입니다."
                    elif key == "event":
                        doc_content = f"이 비디오는 {value} 이벤트의 일부입니다."
                    else:
                        doc_content = f"이 비디오의 {key}는 {value}입니다."
                    
                    memory_doc = Document(
                        page_content=doc_content,
                        metadata={"type": "memory", "key": key, "value": value}
                    )
                    memory_docs.append(memory_doc)
                
                if memory_docs:
                    self.vector_store.add_documents(memory_docs)
                    print(f"DEBUG: Added {len(memory_docs)} existing memory documents to vector store")
            except Exception as e:
                print(f"Warning: Could not add existing memory to vector store: {e}")
        
        # Update global state
        _agent_state["vector_store"] = self.vector_store
        _agent_state["qa_chain"] = self.qa_chain
        _agent_state["video_context"] = self.video_context
        _agent_state["conversation_memory"] = self.conversation_memory
        
        # 5. Save metadata (JSON)
        self.save_metadata()
        
        # Return different messages based on whether vector store was loaded or created
        if vector_store_loaded:
            return "📂 기존 벡터 스토어를 로드했습니다."
        else:
            return "🔧 새로운 벡터 스토어를 생성했습니다."

    def run(self, query):
        """Simple routing-based agent without complex agent framework."""
        if not self.docs:
            return "먼저 비디오를 로드해주세요."
        
        print(f"\n🎯 사용자 질문: {query}")
        
        # Pre-check: answer from memory if applicable
        if self.conversation_memory and any(keyword in query.lower() for keyword in ['뭐', '무엇', '어떤', '누구', '어디', '언제', 'what', 'who', 'which', 'where', 'when', '채널']):
            try:
                print(f"💾 [ROUTE] 저장된 메모리에서 답변 확인 중...")
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
                print(f"💾 [ROUTE] 정보 저장 시도 중...")
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
        
        # Simple routing logic with tool selection display
        query_lower = query.lower()
        
        # Check for specific tool requests
        if any(word in query_lower for word in ['요약', 'summary', 'summarize']):
            print(f"📝 [ROUTE] 비디오 요약 도구 선택")
            return summarize_video.invoke("")
        elif any(word in query_lower for word in ['제목', 'title', 'titles']):
            print(f"🎯 [ROUTE] 제목 생성 도구 선택")
            return generate_titles.invoke("")
        elif any(word in query_lower for word in ['블로그', 'blog', 'post']):
            print(f"✍️ [ROUTE] 블로그 포스트 작성 도구 선택")
            return write_blog_post.invoke("")
        elif any(word in query_lower for word in ['퀴즈', 'quiz', 'test']):
            print(f"❓ [ROUTE] 퀴즈 생성 도구 선택")
            return generate_quiz.invoke("")
        elif any(word in query_lower for word in ['핵심', '중요', 'key', 'moments', 'highlights']):
            print(f"⭐ [ROUTE] 핵심 순간 추출 도구 선택")
            return extract_key_moments.invoke("")
        elif any(word in query_lower for word in ['검색', 'search', '찾아', '기사', '뉴스', '관련', '출처']):
            print(f"🔍 [ROUTE] 웹 검색 도구 선택")
            
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
                result = search_web.invoke(search_query)
                # Save metadata after any interaction
                self.save_metadata()
                return result
            else:
                return "검색할 내용을 구체적으로 알려주세요."
        else:
            # Default to answering questions about the video
            print(f"🤔 [ROUTE] 비디오 질문 답변 도구 선택")
            result = answer_question.invoke(query)
            # Save metadata after any interaction
            self.save_metadata()
            return result
    def add_context_to_vector_store(self, summary):
        """Add context summary to vector store for semantic search."""
        if self.vector_store and summary:
            try:
                context_doc = Document(
                    page_content=f"비디오 요약: {summary}",
                    metadata={"type": "context_summary"}
                )
                self.vector_store.add_documents([context_doc])
                print("DEBUG: Added context summary to vector store")
            except Exception as e:
                print(f"Warning: Could not add context to vector store: {e}")