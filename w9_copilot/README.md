
## Install necessary packages
```
pip install -r requirements.txt
```

## Allow the app to use your gcloud project credentials
```
gcloud auth application-default login
```

## Verify LLM access
```
python scratch.py
```

## Run using streamlit
```
python -m streamlit run w9copilot.py
```

## Navigate to the URL provided by streamlit

http://localhost:8501

## Example questions to ask

* What's a disregarded entity?
* Who's the IRS?
* how many exemptions can i claim?
* what's a federal tax classification?
* What are the possible values?
* 
 