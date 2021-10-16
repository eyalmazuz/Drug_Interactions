import json
import requests

webhook_url = ''

def send_message(text: str):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'content': text})
    requests.post(url=webhook_url, data=payload, headers=headers)
