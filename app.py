import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from typing import Annotated, List

# Load environment variables
load_dotenv()

# Set up Groq client
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="qwen-2.5-32b")

# Define BlogState TypedDict
class BlogState(TypedDict):
    topic: str
    title: str
    blog_content: Annotated[List, add_messages]
    reviewed_content: Annotated[List, add_messages]
    is_blog_ready: str

# Initialize session state
if 'blog_state' not in st.session_state:
    st.session_state.blog_state = None
if 'graph' not in st.session_state:
    st.session_state.graph = None

def init_graph():
    builder = StateGraph(BlogState)
    
    builder.add_node("title_generator", generate_title)
    builder.add_node("content_generator", generate_content)
    builder.add_node("content_reviewer", review_content)
    builder.add_node("quality_check", evaluate_content)
    
    builder.add_edge(START, "title_generator")
    builder.add_edge("title_generator", "content_generator")
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
    
    with st.status("🚀 Generating Titles..."):
        response = llm.invoke(prompt)
        state["title"] = response.content.split("\n")[0].strip('"')
        st.write(f"Selected title: **{state['title']}**")
    return state

def generate_content(state: BlogState):
    prompt = f"""Write a comprehensive blog post titled "{state["title"]}" with:
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
        state["reviewed_content"].append(HumanMessage(content=feedback.content))
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
        state["is_blog_ready"] = "Pass" if "PASS" in verdict else "Fail"
        state["reviewed_content"].append(AIMessage(
            content=f"Verdict: {response.content}"
        ))
        st.write(f"Final Verdict: **{state['is_blog_ready']}**")
    return state

def route_based_on_verdict(state: BlogState):
    return "Pass" if state["is_blog_ready"] == "Pass" else "Fail"

# Streamlit UI components
st.title("AI Blog Generation Assistant")
st.markdown("### Generate high-quality blog posts with AI-powered review process")

topic = st.text_input("Enter your blog topic:", placeholder="Generative AI in Healthcare")
generate_btn = st.button("Generate Blog Post")


if generate_btn and topic:
    st.session_state.graph = init_graph()
    st.session_state.blog_state = BlogState(
        topic=topic,
        title="",
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
    st.subheader("Quality Assurance Report")
    st.write(final_state["reviewed_content"][-1].content)
    # st.write(f"Final Verdict: {final_state['is_blog_ready']}")
    

elif generate_btn and not topic:
    st.error("Please enter a blog topic to get started!")

if st.session_state.blog_state:
    with st.sidebar:
        st.subheader("Generation Details")
        st.write(f"**Topic:** {st.session_state.blog_state['topic']}")
        st.write(f"**Status**: {'✅ Approved' if st.session_state.blog_state['is_blog_ready'] == 'Pass' else '❌ Needs Revision'}")
        st.write(f"**Review Cycles**: {len(st.session_state.blog_state['reviewed_content']) - 1}")
        
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()