import requests
from utils import printResponse

response = requests.post("http://127.0.0.1/api/mgr/signin",
                         data={
                             'username': 'byhy',
                             'password': '99999999'
                         }
                         )
printResponse(response)
