from llm import LLM
from persona import Persona
from typing import Dict, Any

class Person:

    def __init__(self, llm: LLM, persona: Persona, system_prompt: str, history: list[Dict[str,str]]):
        self.llm = llm
        self.persona = persona
        self.system_prompt = system_prompt
        self.history = []

    async def generate(self, prompt: str = None):

        if prompt:
            self.history.append({'role': 'user', "content": prompt })

        if prompt:
            output = await self.llm.generate(prompt, self.system_prompt, self.history)
        else:
            output = await self.llm.generate(self.system_prompt, self.history)
        
        self.history.append({"role": "assistant", "content": output})

        return output

class Conversation:

    def __init__(self, person1: Person, person2: Person):
        self.person1 = person1
        self.person2 = person2

    async def generate(self, rounds: int):

        messages = [await self.person1.generate()]
        
        for _ in range(rounds):
            messages.append(await self.person2.generate(messages[-1]))
            messages.append(await self.person1.generate(messages[-1]))

        return messages