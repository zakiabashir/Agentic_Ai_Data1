from typing import Any
import asyncio
from myconfig.gemini_config import  api_key
from myagent.guardrailinputoutputagent import  general_agent
from agents import (
    Runner,
    
    set_tracing_export_api_key,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    set_tracing_disabled
)
set_tracing_disabled(True) # Disable tracing for better performance

set_tracing_export_api_key(api_key=api_key)


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