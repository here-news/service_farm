"""
Check what type OpenAI returns for embeddings
"""
import asyncio
import os
from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input="Test text for embedding"
    )

    embedding = response.data[0].embedding

    print(f"Type: {type(embedding)}")
    print(f"Length: {len(embedding)}")
    print(f"First 3 values: {embedding[:3]}")
    print(f"Is list: {isinstance(embedding, list)}")
    print(f"String representation: {str(embedding)[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
