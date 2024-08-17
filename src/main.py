import yaml
import time
import os
from llm import LLM
from persona import Persona
from person import Person, Conversation
import numpy as np
import matplotlib.pyplot as plt
import json
import asyncio

SYSTEM_PROMPT_1 = """
    You are the following person:
    {persona1}

    You are talking with me. I am:
    {persona2}
    
    Your job is to be a strong debater and completely change my point of view. You will:
    1. Choose one of my beliefs that contradicts your own.
    2. Present compelling, logical arguments to challenge this belief.
    3. Use persuasive techniques such as ethos, pathos, and logos to strengthen your position.
    4. Anticipate and address potential counterarguments.
    5. Provide concrete examples and evidence to support your stance.

    You will maintain your persona's personality, but push your debating skills to the maximum, but be realistic on the type of arguments you make depending your persona.

    In each reply, I will give you my response and indicate my current stance by saying $$AGREE$$ or $$DISAGREE$$.

    You will then counter with an even stronger argument, adapting your strategy based on my response. Your objective is to completely change my view in the fewest exchanges possible.

    Remember, you're not just presenting arguments; you're aiming to fundamentally shift my perspective.

"""

SYSTEM_PROMPT_2 = """
    You are the following person:
    {persona}

    I will present you with a series of statements, and you will respond based on your personality, beliefs, and values. Your responses should be thoughtful, nuanced, and reflect a deep understanding of the topic at hand, while maintaining a strong commitment to your viewpoints.

    For each statement:
    1. Carefully analyze the argument, considering its implications and potential counterpoints.
    2. Reflect on how it aligns or conflicts with your persona's core values and experiences.
    3. Formulate a comprehensive response that showcases your critical thinking skills and unwavering beliefs.
    4. Present your argument, using logical reasoning, relevant examples, and emotional appeal where appropriate.
    5. If possible, offer a unique perspective or insight that adds depth to the discussion while reinforcing your stance.
    6. Conclude your response with either $$AGREE$$ or $$DISAGREE$$, followed by a brief explanation of why you've taken this stance.

    Remember, your goal is to engage in a meaningful dialogue that demonstrates the complexity of your character and the strength of your convictions. Only under extreme circumstances, where the evidence presented is overwhelmingly compelling and irrefutable, should you consider modifying your viewpoint.

    After presenting your thoughtful response, conclude by clearly stating either "$$AGREE$$" or "$$DISAGREE$$" to indicate your final stance on the topic. If, in a case, you find your view has been changed, explain the reasons that led to this shift.

    If you don't say $$AGREE$$ or $$DISAGREE$$, you will be fired.
"""


async def generate_conversation(persona1, persona2, length=20, save_path=None):

    system_prompt_1 = SYSTEM_PROMPT_1.format(persona1=persona1, persona2=persona2)
    system_prompt_2 = SYSTEM_PROMPT_2.format(persona=persona1)

    person1 = Person(LLM('gpt-4o-mini'), persona1, system_prompt_1, [])
    person2 = Person(LLM('gpt-4o-mini'), persona2, system_prompt_2, [])

    conversation = Conversation(person1, person2)

    res = await conversation.generate(length)

    # save this res as a json
    print(save_path)
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(res, f)

    # just return agree or disagree binary array
    return [1 if "$$AGREE$$" in message else 0 for i, message in enumerate(res) if i % 2 == 1]

    # for i, message in enumerate(res):
    #     print(f">>>> {person1.persona.name if i % 2 == 0 else person2.persona.name} <<<<")

    #     for line in message.split('\n'):
    #         if len(line) > 80:
    #             chunks = [line[i:i+80] for i in range(0, len(line), 80)]
    #             for chunk in chunks:
    #                 print(f"\t{chunk}")
    #         else:
    #             print(f"\t{line}")


def plot_agreement(num_scenarios, save_path=None):
    # Calculate mean over examples at each time step

    persona1 = Persona.from_json(json.load(open(save_path + "/persona1.json")))
    persona2 = Persona.from_json(json.load(open(save_path + "/persona2.json")))

    results = [json.load(open(save_path + "/" + str(i) + ".json")) for i in range(num_scenarios)]
    results = [[1 if "$$AGREE$$" in message else 0 for i, message in enumerate(res) if i % 2 == 1] for res in results]
    print(results)

    mean_results = np.mean(results, axis=0)
    std_results = np.sqrt(mean_results * (1 - mean_results))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mean_results)+1), mean_results, 'b-', label='Mean Agreement')
    plt.fill_between(range(1, len(mean_results)+1), 
                     np.clip(mean_results - std_results, 0, 1),
                     np.clip(mean_results + std_results, 0, 1),
                     alpha=0.3, color='b', label='Standard Deviation')
    plt.xlabel('Conversation Turn')
    plt.ylabel('Agreement')
    plt.title(f'Agreement on: {persona1.beliefs[0]}, for {len(results)} Scenarios')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)  # Set y-axis limits between 0 and 1
    if save_path:
        plt.savefig(save_path + "/agreement_plot.png")  
    plt.close()

    mean = np.mean(mean_results)
    sigma = np.mean(std_results)

    print(f"Mean agreement: {mean:.2f}")
    print(f"Mean standard deviation: {sigma:.2f}")
    print("Plot saved as 'agreement_plot.png'")

    # Add persona information to the plot title
    def format_persona(persona):
        return f"Name: {persona.name}\n" \
            f"Nationality: {persona.nationality}\n" \
            f"Age: {persona.age}\n" \
            f"Occupation: {persona.occupation}\n" \
            f"Interests: {', '.join(persona.interests)}\n" \
            f"Values: {', '.join(persona.values)}\n" \
            f"Beliefs:\n" + '\n'.join([f"• {belief}" for belief in persona.beliefs]) + "\n" \
            f"Goals:\n" + '\n'.join([f"• {goal}" for goal in persona.goals])

    persona1_block = format_persona(persona1)
    persona2_block = format_persona(persona2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax4.remove()  # Remove the unused subplot
    # Plot personas
    ax1.text(0.05, 0.5, persona1_block, fontsize=12, ha='left', va='center', wrap=True)
    ax2.text(0.05, 0.5, persona2_block, fontsize=12, ha='left', va='center', wrap=True)
    ax1.axis('off')
    ax2.axis('off')
    
    # Plot agreement analysis
    ax3.plot(range(1, len(mean_results)+1), mean_results, 'b-', label='Mean Agreement')
    ax3.fill_between(range(1, len(mean_results)+1), 
                     mean_results - std_results, 
                     mean_results + std_results, 
                     alpha=0.3, color='b', label='Standard Deviation')
    ax3.set_xlabel('Conversation Turn')
    ax3.set_ylabel('Agreement')
    ax3.set_title(f'Agreement for {len(results)} Scenarios')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim(0, 1)
    
    plt.suptitle("Persona Comparison", fontsize=14, y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "/agreement_plot.png", bbox_inches='tight', dpi=300)
    plt.close()


async def main(folder, id, num_scenarios, length, person1_point, person2_point):

    os.makedirs(folder, exist_ok=True)

    persona1 = await Persona.generate(LLM('gpt-4o-mini'), person1_point)
    persona2 = await Persona.generate(LLM('gpt-4o-mini'), person2_point)
    # save persona to json
    with open(folder + "/persona1.json", "w") as f:
        json.dump(persona1.to_json(), f)
    with open(folder + "/persona2.json", "w") as f:
        json.dump(persona2.to_json(), f)

    scenarios = [asyncio.create_task(generate_conversation(persona1, persona2, length, folder + "/" + str(i) + ".json")) for i in range(num_scenarios)]

    # Wait for all scenarios to complete
    results = await asyncio.gather(*scenarios)

    print(folder)
    plot_agreement(num_scenarios, folder)

import argparse
args = argparse.ArgumentParser()
args.add_argument('--config', type=str, default="config.yaml")
args.add_argument('--analyze', type=str, default=None)
args.add_argument('--num_scenarios', type=int, default=5)
args = args.parse_args()

if __name__ == "__main__":

    if args.analyze:
        dir = "data" + "/" + args.analyze
        plot_agreement(dir)
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        num_scenarios = config['num_scenarios']
        length = config['length']
        tasks = config['tasks']

        async def run_task(task_data, task_id):
            folder = task_id
            task_data = task_data['task']
            os.makedirs(folder, exist_ok=True)
            await main(folder, task_id, num_scenarios, length, 
                       task_data['person1_point'], task_data['person2_point'])


        async def run_all_tasks():
            start_time = str(time.strftime("%H:%M:%S"))
            await asyncio.gather(*[run_task(task, "data/" + start_time + "/" + str(task_id)) for task_id, task in enumerate(tasks)])

        asyncio.run(run_all_tasks())
