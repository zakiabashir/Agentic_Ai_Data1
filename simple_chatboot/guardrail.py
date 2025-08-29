from typing import Any
from openai import AsyncOpenAI
from dotenv import find_dotenv, load_dotenv
import os
import asyncio

from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
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
set_tracing_disabled(True) # Disable tracing for better performance


load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model_name = os.getenv("GEMINI_MODEL_NAME")


client = AsyncOpenAI(api_key=api_key, base_url=base_url)
model = OpenAIChatCompletionsModel(openai_client=client, model=model_name)

set_tracing_export_api_key(api_key=api_key)


class MathOutPut(BaseModel):
    is_math: bool
    reason: str

@input_guardrail
async def check_input(
    ctx: RunContextWrapper[Any], agent: Agent[Any], input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    input_agent = Agent(
        "InputGuardrailAgent",
        instructions="Check and verify if input is related to math",
        model=model,
        output_type=MathOutPut,
    )
    result = await Runner.run(input_agent, input_data, context=ctx.context)
    final_output = result.final_output

    return GuardrailFunctionOutput(
        output_info=final_output, tripwire_triggered=not final_output.is_math
    )

# --- Output Guardrail Implementation ---
class OutputCheck(BaseModel):
    is_political: bool
    reason: str

@output_guardrail
async def check_output(
    ctx: RunContextWrapper[Any], agent: Agent[Any], output_data: str
) -> GuardrailFunctionOutput:
    # Simple keyword-based check for political content
    keywords = [
        "politician", "prime minister", "president", "politics", "government",
        "minister", "senator", "political party", "election", "parliament"
    ]
    is_political = any(word in output_data.lower() for word in keywords)
    reason = "Output contains political content." if is_political else "Output is safe."
    return GuardrailFunctionOutput(
        output_info=OutputCheck(is_political=is_political, reason=reason),
        tripwire_triggered=is_political
    )
# ---------------------------------------

math_agent = Agent(
    "MathAgent",
    instructions="You are a math agent",
    model=model,
    input_guardrails=[check_input],
)

general_agent = Agent(
    "GeneralAgent",
    instructions="You are a helpful agent",
    model=model,
    output_guardrails=[check_output],  # <-- Output guardrail added here
)

async def main():
    try:
        msg = input("Enter you question : ")
        result = await Runner.run(general_agent, msg)
        print(f"\n\n final output : {result.final_output}")
    except InputGuardrailTripwireTriggered as ex:
        print("Error : invalid prompt")
    except OutputGuardrailTripwireTriggered as ex:
        print("Error: Output blocked due to political content.")

asyncio.run(main())