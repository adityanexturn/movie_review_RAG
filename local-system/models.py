# this is for listing all thew models available in your api 

import requests
api_key = "type your api key here"
url = "https://api.groq.com/openai/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get(url, headers=headers)
print(response.json())
