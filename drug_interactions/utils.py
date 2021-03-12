import json
import requests

webhook_url = 'https://discord.com/api/webhooks/791261360558178324/KgcGEmNWPkM227ZtwWnmEERz0n37Fb642WOgpEWfu5BUan1WhJAVqg95ombetfy6M37y'

def send_message(text: str):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'content': text})
    requests.post(url=webhook_url, data=payload, headers=headers)