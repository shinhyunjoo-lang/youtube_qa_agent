import os
import json
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains.summarize import load_summarize_chain
    from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

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
        self.conversation_memory = {}  # Store user-provided facts

    def load_video(self, url):
        """Loads the video transcript."""
        try:
            # reset state
            self.docs = []
            self.vector_store = None
            self.qa_chain = None
            self.video_context = ""

            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["en", "en-US", "ko"])
            self.docs = loader.load()
            
            if not self.docs:
                return "Error: No transcript found for this video. Please check if the video has English or Korean captions."
                
            # Fallback for title since add_video_info=False doesn't fetch it
            title = self.docs[0].metadata.get('title', 'YouTube Video')
            self.docs[0].metadata['title'] = title
            self.video_context = f"Video Title: {title}"
            
            # Extract video ID for persistence (supports multiple URL formats)
            if "v=" in url:
                # Standard format: youtube.com/watch?v=VIDEO_ID
                self.video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                # Short format: youtu.be/VIDEO_ID
                self.video_id = url.split("youtu.be/")[1].split("?")[0]
            else:
                # Fallback: use hash of URL
                import hashlib
                self.video_id = hashlib.md5(url.encode()).hexdigest()[:12]
                
            return f"Successfully loaded video: {title}"
        except Exception as e:
            return f"Error loading video: {str(e)}"

    def create_vector_store(self):
        """Creates a FAISS vector store, using local persistence."""
        if not self.docs:
            return "No documents to process. Load a video first."
        
        db_path = f"db/{self.video_id}"
        
        # Check if index exists locally
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
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        
        # Generate brief context summary if not already done
        if "Summary:" not in self.video_context:
            try:
                # Get a very brief summary for context (first 3000 chars)
                brief_content = self.docs[0].page_content[:3000]
                summary_prompt = f"In 1-2 sentences, what is this video about?\n\n{brief_content}"
                context_summary = self.llm.invoke([HumanMessage(content=summary_prompt)]).content
                self.video_context += f"\nSummary: {context_summary}"
                print(f"DEBUG: Generated context summary")
            except Exception as e:
                print(f"Warning: Could not generate context summary: {e}")
        
        return "Vector store created/loaded successfully."

    def _get_summary_tool(self):
        """Internal method for summary tool."""
        if not self.docs:
            return "No documents to summarize."
        
        chain = load_summarize_chain(self.llm, chain_type="stuff")
        try:
            return chain.run(self.docs)
        except Exception as e:
            # Fallback for long videos
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            return chain.run(self.docs)

    def _generate_titles_tool(self):
        """Internal method for title generation tool."""
        if not self.docs:
            return "No documents available."
        
        prompt = PromptTemplate(
            template="Based on the following content, suggest 5 catchy YouTube titles:\n\n{text}",
            input_variables=["text"]
        )
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        try:
            return chain.run(self.docs[:1]) 
        except:
             return "Content too long for title generation."

    def _generate_blog_post_tool(self):
        """Internal method for blog post tool."""
        if not self.docs:
            return "No documents available."
        
        prompt = PromptTemplate(
            template="You are a professional technical writer. Create a well-structured blog post based on the following video transcript. Include a catchy title, introduction, key points, and conclusion.\n\n{text}",
            input_variables=["text"]
        )
        try:
            from langchain.chains import LLMChain
        except ImportError:
            from langchain_classic.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        limit = 12000 
        content = self.docs[0].page_content[:limit] 
        return chain.run(content)

    def _generate_quiz_tool(self):
        """Internal method for quiz generation."""
        if not self.docs:
            return "No documents available."
            
        prompt = PromptTemplate(
            template="Based on the following video content, generate a 5-question multiple choice quiz. For each question, provide 4 options and indicate the correct answer.\n\n{text}",
            input_variables=["text"]
        )
        try:
            from langchain.chains import LLMChain
        except ImportError:
            from langchain_classic.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        # Use first chunk or map reduce if needed. For quiz, first chunk might be enough if short, but better to use summary? 
        # using first 12000 chars roughly to capture main content
        return chain.run(self.docs[0].page_content[:12000])

    def _extract_key_moments_tool(self):
        """Internal method for key moments extraction."""
        if not self.docs:
            return "No documents available."
            
        prompt = PromptTemplate(
            template="Identify 5-7 key moments or takeaways from the following video transcript. Use bullet points with a brief description for each.\n\n{text}",
            input_variables=["text"]
        )
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        try:
            return chain.run(self.docs)
        except:
            return "Content too long for key moments extraction. Try summarizing first."

    def _translate_content_tool(self, target_language="Korean"):
        """Internal method for translating summary/content."""
        if not self.docs:
            return "No documents available."
            
        # First get a summary to translate (translating whole transcript is too heavy)
        summary = self._get_summary_tool()
        
        prompt = PromptTemplate(
            template=f"Translate the following text into {target_language}:\n\n{{text}}",
            input_variables=["text"]
        )
        try:
            from langchain.chains import LLMChain
        except ImportError:
            from langchain_classic.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(summary)

    def _web_search_tool(self, query):
        """Internal method for web search."""
        try:
            return self.search.invoke(query)
        except Exception as e:
            return f"Web search failed: {str(e)}"

    def _extract_and_store_memory_tool(self, user_message):
        """Extract and store factual information from user message."""
        try:
            extraction_prompt = f"""Analyze the following user message and extract any factual information they are providing about the video.
Return ONLY a JSON object with key-value pairs. If no factual information is provided, return {{}}.

Examples:
- "이 비디오는 AWS Events 채널에 올라왔어" → {{"channel": "AWS Events"}}
- "발표자는 John이야" → {{"speaker": "John"}}
- "이건 re:Invent 2024 세션이야" → {{"event": "re:Invent 2024"}}
- "Lambda와 DynamoDB를 다루는 내용이야" → {{"topics": ["Lambda", "DynamoDB"]}}
- "내 프로젝트 참고용으로 봤어" → {{"purpose": "프로젝트 참고용"}}

User message: {user_message}

Return JSON:"""
            
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            content = response.content.strip()
            
            # Clean up markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            extracted_data = json.loads(content)
            
            if extracted_data:
                # Update conversation memory
                self.conversation_memory.update(extracted_data)
                stored_keys = ", ".join(extracted_data.keys())
                return f"✓ 정보를 저장했습니다: {stored_keys}"
            else:
                return "저장할 새로운 정보가 없습니다."
                
        except Exception as e:
            return f"메모리 저장 중 오류: {str(e)}"

    def run(self, query):
        """Entry point for the agent. Uses a custom router to select tools."""
        if not self.docs:
             return "Please load a video first."

        # Pre-check: If user is asking about stored information, answer directly from memory
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

        # 1. Select Tool
        # Format memory for display
        memory_str = json.dumps(self.conversation_memory, ensure_ascii=False, indent=2) if self.conversation_memory else "None"
        
        system_prompt = f"""You are a helper agent for a YouTube video Q&A system.
Current Video Context: {self.video_context}
User-Provided Information (Memory): {memory_str}

IMPORTANT: When generating tool_input, ALWAYS use specific information from the Current Video Context above.

You have access to the following tools:
1. Summarizer: use ONLY when the user asks for a general summary, overview, or "what is this video about?". DO NOT use for specific questions.
2. TitleGenerator: Generates catchy titles for the video.
3. BlogWriter: Writes a blog post based on the video.
4. QuizGenerator: Generates a multiple choice quiz based on the video.
5. KeyMoments: Extracts key moments, topics, or takeaways from the video.
6. Translator: Translates the video summary or content into Korean (or user specified language).
7. WebSearch: Searches the web for current events, facts, or information NOT present in the video (e.g., "who is the speaker?", "latest news about X").
   - CRITICAL: For WebSearch, extract specific entities, names, or topics from the Video Context and include them in the search query.
   - BAD: "YouTube Video channel name", "this video topic"
   - GOOD: "[Specific Topic from Summary] latest updates", "[Speaker Name] background", "[Company/Product Name] information"
8. MemoryStore: ONLY when user provides NEW factual information as a STATEMENT (e.g., "이 채널은 AWS Events야", "발표자는 John이야").
   - DO NOT use for questions (e.g., "채널이 뭐야?", "어디 채널이야?")
   - DO NOT use if user is asking about already stored information
   - tool_input should be the user's original message
9. VideoQA: Answers specific questions about the video content, such as "explain the conclusion", "what did the speaker say about X?", or "details about the intro".

CRITICAL DECISION LOGIC:
- If user message is a QUESTION (contains ?, 뭐, 무엇, 어디, 누구, 언제, 왜, 어떻게), DO NOT use MemoryStore
- If user message is a STATEMENT providing new facts, use MemoryStore
- If user is asking about stored information, check User-Provided Information first and answer directly without using tools

Your task is to decide which tool to use based on the user's query.
Return your response in JSON format: {{"tool": "TOOL_NAME", "tool_input": "INPUT_FOR_TOOL"}}
- For Summarizer, TitleGenerator, BlogWriter, QuizGenerator, KeyMoments, Translator: tool_input can be empty or capture specific instructions.
- For WebSearch: tool_input MUST be a specific search query using entities/topics from the Video Context. Never use generic terms like "this video" or "the channel".
- For MemoryStore: tool_input should be the user's original message.
- For VideoQA: tool_input should be a standalone search query optimized to find the answer in the video transcript (e.g., "What is the conclusion of the video?", "Speaker's opinion on AI safety").
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            
            # Simple parsing (robustness can be improved)
            content = response.content
            # Cleanup json potential markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            decision = json.loads(content)
            tool = decision.get("tool")
            tool_input = decision.get("tool_input")
            
            print(f"DEBUG: Agent decided to use tool: {tool}")
            print(f"DEBUG: Tool input: {tool_input}")

            if tool == "Summarizer":
                return self._get_summary_tool()
            elif tool == "TitleGenerator":
                return self._generate_titles_tool()
            elif tool == "BlogWriter":
                return self._generate_blog_post_tool()
            elif tool == "QuizGenerator":
                return self._generate_quiz_tool()
            elif tool == "KeyMoments":
                return self._extract_key_moments_tool()
            elif tool == "Translator":
                return self._translate_content_tool()
            elif tool == "WebSearch":
                return self._web_search_tool(tool_input)
            elif tool == "MemoryStore":
                return self._extract_and_store_memory_tool(query)  # Use original query
            elif tool == "VideoQA":
                if not self.qa_chain:
                     return "Vector store not initialized. Please reload video."
                
                # Fallback to original query if tool_input is empty, though prompt should handle it
                search_query = tool_input if tool_input else query
                return self.qa_chain.run(search_query)
            else:
                # Default to QA if unsure
                if self.qa_chain:
                    return self.qa_chain.run(query)
                return "I'm not sure what tool to use."

        except Exception as e:
            return f"Agent Error: {str(e)}"
