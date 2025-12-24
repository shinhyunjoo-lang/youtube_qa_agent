import os
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

class YouTubeAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.docs = []
        self.vector_store = None
        self.qa_chain = None

    def load_video(self, url):
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["en", "en-US", "ko"])
            self.docs = loader.load()
            
            if not self.docs:
                return "Error: No transcript found for this video. Please check if the video has English or Korean captions."
                
            # Fallback for title since add_video_info=False doesn't fetch it
            title = self.docs[0].metadata.get('title', 'YouTube Video')
            self.docs[0].metadata['title'] = title
            return f"Successfully loaded video: {title}"
        except Exception as e:
            return f"Error loading video: {str(e)}"

    def create_vector_store(self):
        """Creates a FAISS vector store from loaded documents."""
        if not self.docs:
            return "No documents to process. Load a video first."
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(self.docs)
        
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        return "Vector store created successfully."

    def get_summary(self):
        """Generates a summary of the video."""
        if not self.docs:
            return "No documents to summarize."
        
        chain = load_summarize_chain(self.llm, chain_type="stuff")
        try:
            summary = chain.run(self.docs)
            return summary
        except Exception as e:
            # Fallback for long videos (map_reduce)
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            summary = chain.run(self.docs)
            return summary

    def generate_titles(self):
        """Generates catchy YouTube titles."""
        if not self.docs:
            return "No documents available."
        
        prompt = PromptTemplate(
            template="Based on the following content, suggest 5 catchy YouTube titles:\n\n{text}",
            input_variables=["text"]
        )
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
        # We use summarize chain logic but with a custom prompt to treat the whole content
        # Note: For very long videos, this might need map_reduce with a different approach,
        # but 'stuff' works for most standard transcripts.
        try:
            titles = chain.run(self.docs[:1]) # Just use the first part if it's too long, or use map_reduce
            # Better approach: Just use the summary to generate titles to save tokens if generic.
            # But let's try to run on docs.
            return titles
        except:
             return "Content too long for title generation. Try summarizing first."

    def generate_blog_post(self):
        """Generates a blog post based on the video."""
        if not self.docs:
            return "No documents available."
        
        prompt = PromptTemplate(
            template="You are a professional technical writer. Create a well-structured blog post based on the following video transcript. Include a catchy title, introduction, key points, and conclusion.\n\n{text}",
            input_variables=["text"]
        )
        # Using a custom run since load_summarize_chain is specific for summaries
        try:
            from langchain.chains import LLMChain
        except ImportError:
            from langchain_classic.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Naive approach: truncate if too long or use map_reduce summary as input.
        # For simplicity, let's assume we can fit the content or a large chunk of it.
        # If it's too large, we should probably use the summary.
        limit = 12000 # rudimentary char limit
        content = self.docs[0].page_content[:limit] 
        return chain.run(content)

    def answer_question(self, question):
        """Answers a question using the QA chain."""
        if not self.qa_chain:
            return "Please load a video and create the vector store first."
        
        return self.qa_chain.run(question)
