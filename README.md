# Retico-smolAgent

## Overview
This module integrates agentic AI to the Retico framework. This can generate and execute code, using tools on top of powerful LLM. This module can act as separate or combination of all NLU + DM + NLG components

**Agent Module vs LLM Module**
***SmolAgents Module:****

- Capabilities: Text generation + Code execution + Tool usage
- Real-world Actions: ✅ Can perform calculations, API calls, file operations

****Traditional LLM Module:****
- Capabilities: Text generation only
- Real-world Actions: ❌ Text responses only

****Simple Example:**** "What is the weather in Boise, Idaho today?"

- LLM Module: might made it up reponse or "I don't have access to real-time weather data, but you can check weather.com..."
- SmolAgents Module: Actually calls weather API and returns real temperature


## Installation
**Prerequisites**
- Python 3.10.x
- smolagents: 1.19.x


**Setup**

Install SmolAgents Module:
Clone this repository:
```bash
https://github.com/AnhBui1108/retico-smolAgent.git
cd retico-smolAgent
```

## Example:
```bash
import os, sys

import smolagents
from smolagents import CodeAgent, DuckDuckGoSearchTool
from smolagents.models import OpenAIServerModel

prefix = '<path-to-retico-module-repositories>'

os.environ['RETICO'] = prefix +'retico_core'
os.environ['WS']= prefix + 'retico-whisperasr'
os.environ['AG']= prefix + 'retico-smolAgent'
os.environ['TTS'] = prefix + 'retico-speechbraintts'

sys.path.append(os.environ['RETICO'])
sys.path.append(os.environ['WS'])
sys.path.append(os.environ['AG'])
sys.path.append(os.environ['TTS'])


from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_whisperasr.whisperasr import WhisperASRModule
from smolAgents2 import SmolAgentsModule


# System prompt
prompt = """
You are a friendly assistant, Listen to the user request, ask for clarification of you need more information

CRITICAL: Always put what the USER should HEAR in final_answer().
When user provides information, acknowledge it and ask for the next piece of information needed.
When providing numerical answers, always include context words. 
For example, instead of just '10', say 'The answer is ten' or 'I count ten items'.
EXAMPLE FORMAT:
<code>
final_answer("I can see a pair of tortoiseshell glasses and TWO tube of lip gloss on a wooden surface.")
</code>
"""


# Smolagent Model
model = OpenAIServerModel(
    model_id= "qwen/qwen2.5-vl-32b-instruct:free", # You can go wth 
    api_base = "https://openrouter.ai/api/v1",
    stream = False,
    api_key="YOUR-API-KEY")  # Insert your API key here

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    instructions = prompt
)


mic = MicrophoneModule(chunk_size = 320, rate = 16000)
debug = DebugModule()
asr = WhisperASRModule()
dm = SmolAgentsModule(agent)

mic.subscribe(asr)
asr.subscribe(dm)
dm.subscribe(debug)

mic.run()
asr.run()
dm.run()
debug.run()

input()

asr.stop()
mic.stop()
dm.stop()
debug.stop()


