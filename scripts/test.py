import  json
import requests
import pandas as pd

url = "https://admin.opendatanepal.com/api/action/datastore_search?resource_id=918b774b-2a51-404f-92a0-0be1fe780f90&sort=_id asc&limit=100000"


df=pd.read_json("data.json")
print(df.head())