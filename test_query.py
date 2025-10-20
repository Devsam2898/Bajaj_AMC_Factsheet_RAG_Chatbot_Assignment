import requests

url = "https://devsam2898--amc-factsheet-rag-fastapi-app.modal.run/query"
query = {"query": "Give me the fund returns % of last 15 days for Bajaj Finserv Liquid Fund."}

resp = requests.post(url, json=query)
print(resp.status_code)
print(resp.text)
