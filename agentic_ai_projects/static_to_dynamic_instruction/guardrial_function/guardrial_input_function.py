from agents import input_guardrail, RunContextWrapper, GuardrailFunctionOutput, Runner
from my_agent.guardrial_agents import guardrial_agent


@input_guardrail
async def guardrial_input_function(ctx: RunContextWrapper, agent, input):
    result = await Runner.run(guardrial_agent, input=input, context=ctx.context)

    # ✅ Query text ko context me store karna
    if getattr(result.final_output, "query_text", None):
        ctx.context["query_text"] = result.final_output.query_text

    if getattr(result.final_output, "hotel_name", None):
        ctx.context["hotel_name"] = result.final_output.hotel_name

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=getattr(result.final_output, "is_query_about_hotel", True) is False
    )
    # tripwire_triggered=not result.final_output.is_query_about_hotel_sannata