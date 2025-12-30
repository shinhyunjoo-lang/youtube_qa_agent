import gradio as gr
from agent import YouTubeAgent

agent = None

def load_and_initialize(api_key, url):
    global agent
    updates = [gr.update(interactive=False)] * 7 # 7 buttons
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
        enable_updates = [gr.update(interactive=True)] * 7
        return final_status, *enable_updates
    except Exception as e:
         return f"❌ Error: {str(e)}", *updates

# Wrapper functions for tools to handle uninitialized state
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

def translate_content():
    return run_tool("Translate the summary of this video into Korean.")

def search_video_info():
    return run_tool("Identify the main specific topic, entity, or person in this video and search the web for more background information about them.")

def chat_response(message, history):
    if not agent:
        return "⚠️ Please load a video first."
    return agent.run(message)

# Custom CSS for a cleaner look
custom_css = """
#sidebar { background-color: #f7f9fa; padding: 20px; border-right: 1px solid #e5e7eb; }
#tool-btn { margin-bottom: 10px; }
#output-area { border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; background-color: white; min-height: 200px; }
"""

with gr.Blocks(title="YouTube QA Agent", theme=gr.themes.Soft()) as demo:
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
            translate_btn = gr.Button("🇰🇷 Translate to Korean", elem_id="tool-btn")
            search_btn = gr.Button("🔎 Search Info", elem_id="tool-btn")

        # --- Right Main Area ---
        with gr.Column(scale=4):
            gr.Markdown("### 📄 Tool Output")
            # Using Group to frame the output
            with gr.Group():
                tool_output = gr.Markdown("Select a tool from the left sidebar to see results here...", elem_id="output-area")
            
            gr.Markdown("### 💬 Chat with Video")
            chat_interface = gr.ChatInterface(
                fn=chat_response,
                chatbot=gr.Chatbot(height=450),
                textbox=gr.Textbox(placeholder="Ask a question about the video...", container=False, scale=7),
            )

    # --- Interaction Logic ---
    load_btn.click(
        load_and_initialize,
        inputs=[api_key_input, url_input],
        outputs=[status_output, summarize_btn, titles_btn, blog_btn, quiz_btn, moments_btn, translate_btn, search_btn]
    )

    summarize_btn.click(summarize_video, outputs=tool_output)
    titles_btn.click(generate_titles, outputs=tool_output)
    blog_btn.click(generate_blog, outputs=tool_output)
    quiz_btn.click(generate_quiz, outputs=tool_output)
    moments_btn.click(extract_key_moments, outputs=tool_output)
    translate_btn.click(translate_content, outputs=tool_output)
    search_btn.click(search_video_info, outputs=tool_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", css=custom_css)
