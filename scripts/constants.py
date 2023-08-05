import json

with open('data/kaggle.json', 'r') as f:
    kaggle_dict = json.load(f)

# kaggle credentials
KAGGLE_USERNAME = kaggle_dict['username']
KAGGLE_API_KEY = kaggle_dict['key']