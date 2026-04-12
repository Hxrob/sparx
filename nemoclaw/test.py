import asyncio

from nemoclaw import NemoClaw


async def main():
    claw = NemoClaw()
    result = await claw.handle(
        {
            "decision": "categorized",
            "categories": ["Housing"],
            "question": "my landlord hasn't turned on the heat",
            "summary": "Tenant needs heat restored",
            "response": "",
            "confidence": 0.9,
        }
    )
    print(result.source)  # "form_finder"
    print(result.form_finder)  # {form_name, form_url, intent, summary, next_steps}


if __name__ == "__main__":
    asyncio.run(main())
