import requests
from utils import printResponse
import utils
print(dir(utils))
response = requests.post("http://127.0.0.1/api/mgr/signin",
                         data={
                             'username': 'byhy',
                             'password': '88888888'
                         }
                         )
printResponse(response)