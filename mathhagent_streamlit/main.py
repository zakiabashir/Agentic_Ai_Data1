import streamlit as st
import asyncio
from agents import  Runner
from myagent.math_agent import agent

# Set page config
st.set_page_config(
    page_title="Math Tutor",
    page_icon="🧮",
    layout="centered"
)

# Title
st.title("🧮 Math Tutor")
# Simple input box
user_input = st.text_input("Enter your math question:", placeholder="e.g., plus 10 in 100")

# Process button
if st.button("Get Answer"):
    if user_input:
        with st.spinner("Thinking..."):
            # Use asyncio.run() to handle the async operation in Streamlit
            result = asyncio.run(Runner.run(agent, user_input))
            st.markdown("### Answer:")
            st.markdown(result.final_output)
    else:
        st.warning("Please enter a question!")

# Add some helpful information
with st.expander("ℹ️ How to use"):
    st.markdown("""
    **Examples of questions you can ask:**
    - What is 25 plus 17?
    - Solve 15 × 8
    - Explain how to add fractions
    - What is the square root of 144?
    
    The tutor will provide:
    - Step-by-step explanations
    - Clear examples
    - Detailed reasoning
    """) 