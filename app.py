import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_core.messages import AIMessage
from typing import Annotated, List, Dict, Any
from langdetect import detect
from langchain_community.tools.tavily_search import TavilySearchResults

## Langsmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="Blog Generator Agent"

# Define BlogState TypedDict
class BlogState(TypedDict):
    topic: str
    title: str
    search_results: Annotated[List[Dict[str, Any]], add_messages]
    blog_content: Annotated[List, add_messages]
    reviewed_content: Annotated[List, add_messages]
    is_blog_ready: str

# Initialize session state
if 'blog_state' not in st.session_state:
    st.session_state.blog_state = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'graph_image' not in st.session_state:
    st.session_state.graph_image = None

# Helper function to detect English language
def is_english(text):
    # Ensure we have enough text to analyze
    if not text or len(text.strip()) < 50:
        return False
        
    try:
        # Try primary language detection
        return detect(text) == 'en'
    except:
        # If detection fails, use a more robust approach
        common_english_words = ['the', 'and', 'in', 'to', 'of', 'is', 'for', 'with', 'on', 'that', 
                              'this', 'are', 'was', 'be', 'have', 'it', 'not', 'they', 'by', 'from']
        text_lower = text.lower()
        # Count occurrences of common English words
        english_word_count = sum(1 for word in common_english_words if f" {word} " in f" {text_lower} ")
        # Calculate ratio of English words to text length
        text_words = len(text_lower.split())
        if text_words == 0:  # Avoid division by zero
            return False
            
        english_ratio = english_word_count / min(20, text_words)  # Cap at 20 to avoid skew
        return english_word_count >= 5 or english_ratio > 0.25  # More stringent criteria
    
def init_graph(api_key: str):
    
    global llm
    llm  = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key)
    
    builder = StateGraph(BlogState)
    
    builder.add_node("title_generator", generate_title) ## Generate Title
    builder.add_node("search_web", search_web) ## Search Web using Tavily based in the topic
    builder.add_node("content_generator", generate_content) ## Generate Content using the output of title_generator and search_web
    builder.add_node("content_reviewer", review_content) ## Review Content and generate feedback
    builder.add_node("quality_check", evaluate_content) ## Validate the content based on feedback and generate verdict

    builder.add_edge(START, "title_generator")
    builder.add_edge(START, "search_web")
    builder.add_edge("title_generator", "content_generator")
    builder.add_edge("search_web", "content_generator")
    builder.add_edge("content_generator", "content_reviewer")
    builder.add_edge("content_reviewer", "quality_check")
    
    builder.add_conditional_edges(
        "quality_check",
        route_based_on_verdict,
        {"Pass": END, "Fail": "content_generator"}
    )
    return builder.compile()

# Node functions with state management
def generate_title(state: BlogState):
    prompt = f"""Generate compelling blog title options about {state["topic"]} that are:
    - SEO-friendly
    - Attention-grabbing
    - Between 6-12 words"""
    
    
    response = llm.invoke(prompt)
    state["title"] = response.content.split("\n")[0].strip('"')
    return state

def search_web(state: BlogState):
    
    search_tool = TavilySearchResults(max_results=2)
    
    # Create search query with date to get recent news
    query = f"Latest data on {state['topic']}"
    
    # Execute search
    search_results = search_tool.invoke({"query": query})
    
    # Filter out YouTube results and non-English content
    filtered_results = []
    for result in search_results:
        if "youtube.com" not in result.get("url", "").lower():
            # Check if content is in English
            content = result.get("content", "") + " " + result.get("title", "")
            if is_english(content):
                filtered_results.append(result)
   
    return {
    "search_results": [
        {
            "role": "system",
            "content": f"{result['title']}\n{result['content']}\n(Source: {result['url']})"
        }
        for result in filtered_results
        ]
    }


def generate_content(state: BlogState):
    prompt = f"""Write a comprehensive blog post titled "{state['title']}" and based on the web search results {state['search_results']} with:
    1. Engaging introduction with hook
    2. 3-5 subheadings with detailed content
    3. Practical examples/statistics
    4. Clear transitions between sections
    5. Actionable conclusion
    Style: Professional yet conversational (Flesch-Kincaid 60-70). Use markdown formatting"""
    
    with st.status("📝 Generating Content..."):
        response = llm.invoke(prompt)
        state["blog_content"].append(AIMessage(content=response.content))
        st.markdown(response.content)
    return state

def review_content(state: BlogState):
    content = state["blog_content"][-1].content
    prompt = f"""Critically review this blog content:
    - Clarity & Structure
    - Grammar & Style
    - SEO optimization
    - Reader engagement
    Provide specific improvement suggestions. Content:\n{content}"""
    
    with st.status("🔍 Reviewing Content..."):
        feedback = llm.invoke(prompt)
        state["reviewed_content"].append(AIMessage(content=feedback.content))
        st.write(feedback.content)
    return state

def evaluate_content(state: BlogState):
    content = state["blog_content"][-1].content
    feedback = state["reviewed_content"][-1].content
    
    prompt = f"""Evaluate blog content against editorial feedback (Pass/Fail):
    Content: {content}
    Feedback: {feedback}
    Answer only Pass or Fail:"""
    
    with st.status("✅ Evaluating Quality..."):
        response = llm.invoke(prompt)
        verdict = response.content.strip().upper()
        state["is_blog_ready"] = "Pass" if "PASS" or "Pass" in verdict else "Fail"
        state["reviewed_content"].append(AIMessage(
            content=f"Verdict: {response.content}"
        ))
        st.write(f"Final Verdict: **{state['is_blog_ready']}**")
    return state

def route_based_on_verdict(state: BlogState):
    return "Pass" if state["is_blog_ready"] == "Pass" else "Fail"

# Streamlit UI components
st.title("🚀 BlogForge Pro Agent")
st.markdown("""
**Smart Blog Generation with Auto-Refinement**  
*From first draft to final edit - AI-assisted writing meets professional standards*  
✓ SEO-Optimized Structure ✓ Grammar & Style Checks ✓ Feedback-Driven Revisions  
""")

# Sidebar components
with st.sidebar:

    st.subheader("Configuration")
        
    # Groq API Key Input
    api_key = st.text_input("Groq API Key:", 
                          type="password",
                          value=os.getenv("GROQ_API_KEY", ""))
    
     # Validate API key
    if not api_key:
        st.warning("⚠️ Please enter your GROQ API key to proceed. Don't have? refer : https://console.groq.com/keys ")
        
    # Groq API Key Input
    tavily_api_key = os.environ["TAVILY_API_KEY"] =  st.session_state["TAVILY_API_KEY"] = st.text_input("Tavily API Key:", 
                          type="password",
                          value=os.getenv("TAVILY_API_KEY", ""))
    
    # Validate API key
    if not tavily_api_key:
        st.warning("⚠️ Please enter your TAVILY_API_KEY key to proceed. Don't have? refer : https://app.tavily.com/home")
                    
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()
        
        
    st.subheader("Workflow Overview")
    st.image("workflow_graph.png")
        
# Main content
topic = st.text_input("Enter your blog topic:", placeholder="Generative AI in Healthcare")
generate_btn = st.button("Generate Blog Post")

if generate_btn:
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar!")
        st.stop()
    
    if not topic:
        st.error("Please enter a blog topic!")
        st.stop()
    
    try:
        
        # Initialize and run graph
        st.session_state.graph = init_graph(api_key)
        
        st.session_state.blog_state = BlogState(
            topic=topic,
            title="",
            search_results=[],
            blog_content=[],
            reviewed_content=[],
            is_blog_ready=""
        )
        
        # Execute the graph
        final_state = st.session_state.graph.invoke(st.session_state.blog_state)
        st.session_state.blog_state = final_state
        
        
        # Display results
        st.success("Blog post generation complete!")
        st.markdown("---")
        st.subheader("Final Blog Post")
        st.markdown(final_state["blog_content"][-1].content)
        
        st.markdown("---")
        st.subheader("Generated Title")
        st.write(final_state["title"])
        
        st.markdown("---")
        st.subheader("Web Search Results")
        st.write(final_state["search_results"][-1].content)
        
        st.markdown("---")
        st.subheader("Quality Assurance Report")
        st.write(final_state["reviewed_content"][-1].content)
        
        if st.session_state.blog_state:
            st.markdown("---")
            st.subheader("Generation Status")
            st.write(f"**Topic:** {st.session_state.blog_state['topic']}")
            st.write(f"**Status**: {'✅ Approved' if st.session_state.blog_state['is_blog_ready'] == 'Pass' else '❌ Needs Revision'}")
            st.write(f"**Review Cycles**: {len(st.session_state.blog_state['reviewed_content']) - 1}")
        
    except Exception as e:
        st.error(f"Error in blog generation: {str(e)}")