Accompanies the blog post:

PydanticAI: an OSS framework that supports evaluation-driven development of model-agnostic agents

https://medium.com/@lakshmanok/evaluation-driven-development-for-agentic-applications-using-pydanticai-d9293ac81d91

Instructions to run the PydanticAI tutorial:

1. In your favorite Python IDE, git clone this repository
2. Install requirements: ```pip install -r requirements.txt```
3. Edit wikipedia_tool.py line 47 from ```real_or_fake = "Fake"``` to ```real_or_fake = "Real"```
4. Run the wikipedia tool to create cache file: ```python wikipedia_tool.py```
5. Edit wikipedia_tool.py back to Fake: ```real_or_fake = "Fake"```
6. Now, run each of the examples. For example: ```python 1_zero_shot.py```
