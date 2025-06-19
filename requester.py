import requests

url = "http://127.0.0.1:8000/submit/"



data = {"input_string": "Find vegan cafes near Bondi"}

response = requests.post(url, json=data)
print(response.json())