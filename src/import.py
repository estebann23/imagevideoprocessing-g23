import kagglehub
import os

KAGGLE_API_TOKEN="KGAT_6716038ad78e3e6d9f96899d00f51e87"

username="estebanna"
key="156af646e15dabadbf85a3eb37f35c36"

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

# Download latest version
path = kagglehub.competition_download('iivp-2026-challenge')

print("Path to competition files:", path)