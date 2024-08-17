import asyncio
from litellm import acompletion

class LLM:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    async def generate(self, prompt, system_prompt = None, history = []):

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        response = await acompletion(
            model=self.model,
            messages=messages
        )
        
        return response.choices[0].message.content



