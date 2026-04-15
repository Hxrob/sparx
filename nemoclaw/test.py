"""
Smoke test for NemoClaw orchestrator.

Run with:
    pytest nemoclaw/test.py -v

Requires the LLM server to be running at SPARX_LLM_CHAT_URL (default: http://localhost:8081).
"""
import asyncio
import pytest

from nemoclaw import NemoClaw, NemoClawResult


SAMPLE_DIRECTION = {
    "decision": "categorized",
    "categories": ["Housing"],
    "question": "my landlord hasn't turned on the heat",
    "summary": "Tenant needs heat restored",
    "response": "",
    "confidence": 0.9,
}


@pytest.mark.asyncio
async def test_nemoclaw_returns_result():
    claw = NemoClaw()
    result = await claw.handle(SAMPLE_DIRECTION)

    assert isinstance(result, NemoClawResult), "Expected a NemoClawResult"
    assert result.source in ("open_data", "form_finder", "form_finder+open_data", "none")
    assert result.decision == "categorized"
    assert "Housing" in result.categories


@pytest.mark.asyncio
async def test_nemoclaw_irrelevant_passthrough():
    """Irrelevant decisions should pass through without hitting Open Data or FormFinder."""
    claw = NemoClaw()
    result = await claw.handle({
        "decision": "irrelevant",
        "categories": [],
        "question": "",
        "summary": "",
        "response": "OK",
        "confidence": 1.0,
    })

    assert result.source == "none"
    assert result.open_data is None
    assert result.form_finder is None


if __name__ == "__main__":
    # quick manual smoke test without pytest
    async def _run():
        claw = NemoClaw()
        result = await claw.handle(SAMPLE_DIRECTION)
        print(f"source:      {result.source}")
        print(f"form_finder: {result.form_finder}")
        print(f"open_data:   {result.open_data}")
        print(f"response:    {result.response}")

    asyncio.run(_run())
