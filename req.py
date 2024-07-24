import requests

url = "https://a174-18-233-169-33.ngrok-free.app/process"
input_text = "Our story begins in a peaceful woodland kingdom where a lively squirrel named Frolic made his abode high up within a cedar tree's embrace."

response = requests.post(url, data={'input_text': input_text})

try:
    response_data = response.json()
    print(response_data)
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON response. Response content:", response.content)