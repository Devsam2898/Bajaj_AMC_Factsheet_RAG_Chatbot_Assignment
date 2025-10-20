import requests

url = "https://devsam2898--amc-factsheet-rag-fastapi-app.modal.run/ingest"
file_path = r"D:\Bajaj\data\amc_factsheets\bajaj_finserv_factsheet.pdf"

with open(file_path, "rb") as f:
    files = {"file": f}
    resp = requests.post(url, files=files)

print(resp.status_code)
print(resp.text)
