import gradio as gr
from agent import YouTubeAgent

agent = None

def load_and_initialize(api_key, url):
    global agent
    if not api_key:
        return "⚠️ Please enter an OpenAI API Key.", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    
    # Initialize basic agent just to check API key presence, logic handling in agent
    try:
        agent = YouTubeAgent(api_key)
        status = agent.load_video(url)
        
        if "Error" in status:
            return f"❌ {status}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        
        # Auto-create vector store
        vs_status = agent.create_vector_store()
        
        # Success message
        final_status = f"✅ Video Loaded!\n{vs_status}"
        return final_status, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
         return f"❌ Error: {str(e)}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

# Wrapper functions for tools to handle uninitialized state
def run_tool(tool_function):
    if not agent:
        return "⚠️ Agent not initialized. Please load a video first."
    return tool_function()

def summarize_video():
    return run_tool(lambda: agent.get_summary())

def generate_titles():
    return run_tool(lambda: agent.generate_titles())

def generate_blog():
    return run_tool(lambda: agent.generate_blog_post())

def chat_response(message, history):
    if not agent:
        return "⚠️ Please load a video first."
    return agent.answer_question(message)

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
        outputs=[status_output, summarize_btn, titles_btn, blog_btn]
    )

    summarize_btn.click(summarize_video, outputs=tool_output)
    titles_btn.click(generate_titles, outputs=tool_output)
    blog_btn.click(generate_blog, outputs=tool_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", css=custom_css)
