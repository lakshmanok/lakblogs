from unittest.mock import patch, Mock
import json

from deepeval import evaluate
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase

import lg_weather_agent

def write_mock_data(filename="chicago_weather.json"):
    data = retrieve_weather_data(41.8781, -87.6298)
    with open(filename, "w") as ofp:
        json.dump(data, ofp)
    print(f"Wrote {filename}")

# write_mock_data()

# read the hardcoded data, and use it as the return value for the function under test
with open("chicago_weather.json", "r") as ifp:
    chicago_weather = json.load(ifp)
   
@patch('lg_weather_agent.retrieve_weather_data', Mock(return_value=chicago_weather))
def mock_retrieve_weather_data():
    data = lg_weather_agent.retrieve_weather_data(41.8781, -87.6298)
    print(data)

# mock_retrieve_weather_data()

app = lg_weather_agent.create_app()

@patch('lg_weather_agent.retrieve_weather_data', Mock(return_value=chicago_weather))
def run_query_with_mock():
    result = lg_weather_agent.run_query(app, "Is it raining in Chicago?")
    print(result[-1])

#run_query_with_mock()


@patch('lg_weather_agent.retrieve_weather_data', Mock(return_value=chicago_weather))
def test_query_rain_today():
    input_query = "Is it raining in Chicago?"
    expected_output = "No, it is not raining in Chicago right now."
    result = lg_weather_agent.run_query(app, "Is it raining in Chicago?")
    actual_output = result[-1]
    retrieval_context = json.dumps(weather_data)
    
    # use RAGAS
    metric = RagasMetric(threshold=0.5, model='gemini-1.5-flash')
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    metric.measure(test_case)
    print(metric.score)

test_query_rain_today()
