import json
import requests

webhook_url = 'https://discord.com/api/webhooks/831222360102142052/oiWQffWWECZCFDIDylr9hZuAV909DXifDX0WKZWTinq6tS8LDamfD2PjsuDekIkUIf6y'

def send_message(text: str):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'content': text})
    requests.post(url=webhook_url, data=payload, headers=headers)
