import requests
from utils import printResponse


# 创建 Session 对象
s = requests.Session()

# 通过 Session 对象 发送请求
response = s.post("http://127.0.0.1/api/mgr/signin",
       data={
           'username': 'byhy',
           'password': '88888888'
       })

# 通过 Session 对象 发送请求
response = s.get("http://127.0.0.1/api/mgr/customers",
      params={
          'action'    :  'list_customer',
          'pagesize'  :  10,
          'pagenum'   :  1,
          'keywords'  :  '',
      })

printResponse(response)
