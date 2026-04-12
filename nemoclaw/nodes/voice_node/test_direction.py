"""
Quick test for the direction engine.
No Parakeet, no server — just feeds transcripts into the direction LLM (SPARX_LLM_CHAT_URL).

Run:
  cd nodes/voice_node
  python test_direction.py
"""
import asyncio
import direction_engine

TRANSCRIPTS = [
    "I lost my house last month and I have a six year old",
    "I haven't been able to buy food this week, I'm really struggling",
    "How are you doing today?",
    "I got fired and I don't know how to pay rent",
    "My electricity got shut off",
    "I need help with my immigration papers",
]

async def main():
    for transcript in TRANSCRIPTS:
        print(f"\nTranscript: \"{transcript}\"")
        result = await direction_engine.process(transcript)
        print(f"  Decision:    {result.decision.value}")
        print(f"  Categories:  {', '.join(c.value for c in result.categories) if result.categories else 'none'}")
        print(f"  Question:    {result.question}")
        print(f"  Summary:     {result.summary}")
        print(f"  Response:    {result.response}")
        print(f"  Confidence:  {result.confidence:.0%}")
        if result.missing_info:
            print(f"  Missing:     {', '.join(result.missing_info)}")

asyncio.run(main())
