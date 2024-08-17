import asyncio
import json
from llm import LLM
from pydantic import BaseModel


class Persona(BaseModel):
    name: str
    nationality: str
    age: int
    gender: str
    occupation: str
    interests: list[str]
    values: list[str]
    beliefs: list[str]
    goals: list[str]

    def to_dict(self) -> dict:
        """
        Convert the Persona instance to a dictionary for JSON serialization.
        """
        return {
            "name": self.name,
            "nationality": self.nationality,
            "age": self.age,
            "gender": self.gender,
            "occupation": self.occupation,
            "interests": self.interests,
            "values": self.values,
            "beliefs": self.beliefs,
            "goals": self.goals
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Persona instance from a dictionary.
        """
        return cls(**data)

    def to_json(self) -> str:
        """
        Convert the Persona instance to a JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """
        Create a Persona instance from a JSON string.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    async def generate(cls, llm: LLM, persona_seed: str):
        prompt = f"""Generate a persona with the following attributes: 
            "name, age, gender, occupation, interests, values, beliefs, and goals. "
            "Provide the information in a structured format."
            
            Here are two examples:

            Example 1:
            (nationality:Chinese)
            (name:Emily Chen)
            (age:28)
            (gender:Female)
            (occupation:Software Engineer)
            (interests:Hiking, photography, machine learning)
            (values:Honesty, creativity, continuous learning)
            (beliefs:Technology can solve many global problems)
            (goals:Start a tech company, travel to 50 countries, learn three new programming languages)
            
            Example 2:
            (nationality:American)
            (name:Marcus Johnson)
            (age:42)
            (gender:Male)
            (occupation:High School Teacher)
            (interests:Classic literature, gardening, jazz music)
            (values:Education, community service, environmental sustainability)
            (beliefs:Every student has potential)
            (goals:Write a novel, start a community garden project, earn a PhD in Education)
            
            Now, generate a new, unique persona following this format and this persona {persona_seed}

            """
            
        response = await llm.generate(prompt)
        
        # Parse the response and create a Persona instance
        persona_data = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = map(str.strip, line.strip('()').replace('(', '').replace(')', '').split(':', 1))
                key = key.lower()
                if key in ['interests', 'values', 'beliefs', 'goals']:
                    persona_data[key] = [item.strip() for item in value.split(',')]
                elif key == 'age':
                    try:
                        persona_data[key] = int(value.rstrip(')'))
                    except ValueError:
                        raise ValueError(f"Invalid age value: {value}")
                else:
                    persona_data[key] = value

        # Ensure all required fields are present
        required_fields = ['name', 'nationality', 'age', 'gender', 'occupation', 'interests', 'values', 'beliefs', 'goals']
        for field in required_fields:
            if field not in persona_data:
                raise ValueError(f"Missing required field: {field}")

        # Convert string values to appropriate types
        for list_field in ['interests', 'values', 'beliefs', 'goals']:
            if isinstance(persona_data[list_field], str):
                persona_data[list_field] = [item.strip() for item in persona_data[list_field].split(',')]

        return cls(**persona_data)


if __name__ == "__main__":
    async def main():
        llm = LLM()
        persona = await Persona.generate(llm, "An expert negotiator")
        print(persona)
        
        # Test JSON serialization and deserialization
        json_str = persona.to_json()
        print("JSON representation:")
        print(json_str)
        
        deserialized_persona = Persona.from_json(json_str)
        print("Deserialized persona:")
        print(deserialized_persona)

    asyncio.run(main())