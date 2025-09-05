from typing import Any
from openai import AsyncOpenAI
from dotenv import find_dotenv, load_dotenv
import os
import chainlit as cl
from dataschema.mydataschema import OutputCheck
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    OpenAIChatCompletionsModel,
    TResponseInputItem,
    input_guardrail,
    set_tracing_export_api_key,
    InputGuardrailTripwireTriggered,
    output_guardrail,
    OutputGuardrailTripwireTriggered,
    set_tracing_disabled
)

# 👇 import your custom agent
from agent.myagent import agent  

set_tracing_disabled(True)  # Disable tracing for better performance

load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model_name = os.getenv("GEMINI_MODEL_NAME")

client = AsyncOpenAI(api_key=api_key, base_url=base_url)
model = OpenAIChatCompletionsModel(openai_client=client, model=model_name)

set_tracing_export_api_key(api_key=api_key)

# -------- Guardrails --------
# Input guardrail to block Indian-related content
# Output guardrail to block political content
# ---------------------------------------

@input_guardrail
async def check_input(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Block if input contains Indian-related content."""
    user_text = input_data if isinstance(input_data, str) else str(input_data)

    indian_keywords = [
        "india", "indian", "delhi", "mumbai", "bollywood", "hindi", "bharat", "modi"
    ]

    contains_indian = any(word in user_text.lower() for word in indian_keywords)
    reason = "Input contains Indian content." if contains_indian else "Input is safe."

    return GuardrailFunctionOutput(
        output_info={"is_indian": contains_indian, "reason": reason},
        tripwire_triggered=contains_indian
    )


@output_guardrail
async def check_output(
    ctx: RunContextWrapper[Any], agent: Agent[Any], output_data: str
) -> GuardrailFunctionOutput:
    political_keywords = [
        "politician", "prime minister", "president", "politics", "government",
        "minister", "senator", "political party", "election", "parliament"
    ]
    is_political = any(word in output_data.lower() for word in political_keywords)
    reason = "Output contains political content." if is_political else "Output is safe."
    return GuardrailFunctionOutput(
        output_info=OutputCheck(is_political=is_political, reason=reason),
        tripwire_triggered=is_political
    )
# ---------------------------------------

general_agent = Agent(
    "GeneralAgent",
    instructions="You are a helpful agent",
    model=model,
    input_guardrails=[check_input],   # ✅ input guardrail added
    output_guardrails=[check_output]  # ✅ output guardrail added
)

# -------- Conversation History --------
@cl.on_chat_start
async def start_chat():
    cl.user_session.set("history", [])  # empty history list

# -------- Chainlit UI --------
@cl.on_message
async def handle_message(message: cl.Message):
    try:
        history = cl.user_session.get("history", [])

        # By default use general agent
        agent_to_use = general_agent  

        # Agar user "custom" likhe to aapka my_agent chale
        if "custom" in message.content.lower():
            agent_to_use = agent  

        # Append user query to history
        history.append({"role": "user", "content": message.content})

        # Run agent
        result = await Runner.run(agent_to_use, message.content)

        # Append assistant response to history
        history.append({"role": "assistant", "content": result.final_output})
        cl.user_session.set("history", history)

        # Send reply
        await cl.Message(content=f"🤖 {result.final_output}").send()

    except InputGuardrailTripwireTriggered:
        await cl.Message(content="🚫 Indian content blocked").send()
    except OutputGuardrailTripwireTriggered:
        await cl.Message(content="❌ Output blocked: political content detected.").send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()
