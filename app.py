import gradio as gr
from agent import YouTubeAgent
import time

agent = None

def load_and_initialize(api_key, url):
    global agent
    updates = [gr.update(interactive=False)] * 6 # 6 buttons
    if not api_key:
        return "⚠️ Please enter an OpenAI API Key.", *updates
    
    # Initialize basic agent just to check API key presence, logic handling in agent
    try:
        agent = YouTubeAgent(api_key)
        status = agent.load_video(url)
        
        if "Error" in status:
             return f"❌ {status}", *updates
        
        # Auto-create vector store
        vs_status = agent.create_vector_store()
        
        # Success message
        final_status = f"✅ Video Loaded!\n{vs_status}"
        enable_updates = [gr.update(interactive=True)] * 6
        return final_status, *enable_updates
    except Exception as e:
         return f"❌ Error: {str(e)}", *updates

# Wrapper functions for tools to handle uninitialized state
def run_tool(prompt):
    if not agent:
        return "⚠️ Agent not initialized. Please load a video first."
    return agent.run(prompt)

def summarize_video():
    return run_tool("Please summarize this video.")

def generate_titles():
    return run_tool("Generate catchy titles for this video.")

def generate_blog():
    return run_tool("Write a blog post for this video.")

def generate_quiz():
    return run_tool("Create a quiz for this video.")

def extract_key_moments():
    return run_tool("Extract key moments from this video.")

def search_video_info():
    return run_tool("Identify the main specific topic, entity, or person in this video and search the web for more background information about them.")

def chat_response_streaming(message, history):
    """Streaming chat response function"""
    if not agent:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please load a video first."})
        yield history, "", gr.update(visible=False)
        return
    
    # Special handling for Rick and Morty easter egg
    if "릭앤모티" in message.lower() or "릭 앤 모티" in message.lower() or "rick and morty" in message.lower():
        import os
        response_text = """🌌 **Rick and Morty Easter Egg 발견!** 🌌"""
        
        # Add user message first
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield history, "", gr.update(visible=False)
        
        # Stream the response character by character
        partial_response = ""
        for char in response_text:
            partial_response += char
            history[-1]["content"] = partial_response
            yield history, "", gr.update(visible=False)
            time.sleep(0.02)  # Small delay for streaming effect
        
        # Show image if exists
        if os.path.exists("rick_morty.png"):
            yield history, "", gr.update(visible=True, value="rick_morty.png")
        else:
            yield history, "", gr.update(visible=False)
        return
    
    # Add user message first
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, "", gr.update(visible=False)
    
    # Get response from agent
    response = agent.run(message)
    
    # Stream the response while preserving line breaks
    import re
    # Split by words but preserve line breaks and spaces
    tokens = re.findall(r'\S+|\s+', response)
    partial_response = ""
    
    for token in tokens:
        partial_response += token
        history[-1]["content"] = partial_response
        yield history, "", gr.update(visible=False)
        # Only add delay for actual words, not spaces/newlines
        if token.strip():
            time.sleep(0.03)

# Custom CSS for a cleaner look
custom_css = """
#sidebar { background-color: #f7f9fa; padding: 20px; border-right: 1px solid #e5e7eb; }
#tool-btn { margin-bottom: 10px; }
#output-area { border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; background-color: white; min-height: 200px; }
"""

with gr.Blocks(title="YouTube QA Agent") as demo:
    gr.Markdown("# 📺 YouTube QA & Content Repurposing Agent")
    
    with gr.Row():
        # --- Left Sidebar ---
        with gr.Column(scale=1, elem_id="sidebar", min_width=300):
            gr.Markdown("### ⚙️ Configuration")
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...", lines=1)
            url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...", lines=1)
            load_btn = gr.Button("🚀 Load Video", variant="primary")
            
            status_output = gr.Textbox(label="Status", interactive=False, lines=3)
            
            gr.Markdown("---")
            gr.Markdown("### 🛠️ Creator Tools")
            summarize_btn = gr.Button("📝 Summarize Video", elem_id="tool-btn")
            titles_btn = gr.Button("💡 Generate Titles", elem_id="tool-btn")
            blog_btn = gr.Button("✍️ Write Blog Post", elem_id="tool-btn")
            quiz_btn = gr.Button("🎓 Generate Quiz", elem_id="tool-btn")
            moments_btn = gr.Button("⏱️ Key Moments", elem_id="tool-btn")
            search_btn = gr.Button("🔎 Search Info", elem_id="tool-btn")

        # --- Right Main Area ---
        with gr.Column(scale=4):
            gr.Markdown("### 📄 Tool Output")
            # Using Group to frame the output
            with gr.Group():
                tool_output = gr.Markdown("Select a tool from the left sidebar to see results here...", elem_id="output-area")
            
            gr.Markdown("### 💬 Chat with Video")
            
            # Custom chat interface that can handle images
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=600, label="Chat History")
                with gr.Column(scale=1):
                    # Image display area for Rick and Morty easter egg
                    easter_egg_image = gr.Image(visible=False, label="Easter Egg!")
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask a question about the video...", container=False, scale=7, label="Message")
                send_btn = gr.Button("Send", scale=1)

    # --- Interaction Logic ---
    load_btn.click(
        load_and_initialize,
        inputs=[api_key_input, url_input],
        outputs=[status_output, summarize_btn, titles_btn, blog_btn, quiz_btn, moments_btn, search_btn]
    )

    summarize_btn.click(summarize_video, outputs=tool_output)
    titles_btn.click(generate_titles, outputs=tool_output)
    blog_btn.click(generate_blog, outputs=tool_output)
    quiz_btn.click(generate_quiz, outputs=tool_output)
    moments_btn.click(extract_key_moments, outputs=tool_output)
    search_btn.click(search_video_info, outputs=tool_output)
    
    # Chat interface event handlers
    send_btn.click(
        chat_response_streaming,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, easter_egg_image]
    )
    
    msg.submit(
        chat_response_streaming,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, easter_egg_image]
    )

if __name__ == "__main__":
    #demo.launch(server_name="0.0.0.0", css=custom_css)
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, theme=gr.themes.Soft())
