## Example of building an assistant with DSPy

What's being demonstrated is a bidding assistant for the game of bridge.
You don't need to understand bridge to understand the concepts here.
The reason I chose bridge is that there is a lot of jargon, human judgement involved,
and several external tools that an advisor can use.  Kind of like an industry problem.
But because it's a game, there is no confidential information involved.

Accompanies Medium article [Building an AI Assistant with DSPy: A way to program and tune prompt-agnostic LLM agent pipelines](https://lakshmanok.medium.com/building-an-ai-assistant-with-dspy-2e1e749a1a95)

### How to build
Clone this repo:
```commandline
git clone https://github.com/lakshmanok/lakblogs
```
Change to the directory
```commandline
cd lakblogs/bridge_bidding_advisor
```
Install the requirements:
```commandline
pip install -r requirements.txt
```
Create a file named api_keys.env and add one/both of the following lines:
```commandline
GOOGLE_API_KEY=AI...
OPENAI_API_KEY=sk-...
```
Edit the file bidding_advisor.py to initialize the appropriate API and comment out the other:
```commandline
    dspy_init.init_gemini_pro(temperature=0.0)
    # dspy_init.init_gpt35(temperature=0.0)
```
Index the bridege bidding system into Chroma DB:
```commandline
python index_bidding_system.py
```
Try it out:
```commandline
python bidding_advisor.py
```
