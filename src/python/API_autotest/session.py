import requests


# 打印HTTP响应消息的函数
def printResponse(response):
    print('\n\n-------- HTTP response * begin -------')
    print(response.status_code)

    for k, v in response.headers.items():
        print(f'{k}: {v}')

    print('')

    print(response.content.decode('utf8'))
    print('-------- HTTP response * end -------\n\n')


# 创建 Session 对象
s = requests.Session()

# 通过 Session 对象 发送请求
response = s.post("http://127.0.0.1/api/mgr/signin",
       data={
           'username': 'byhy',
           'password': '88888888'
       })

printResponse(response)

# 通过 Session 对象 发送请求
response = s.get("http://127.0.0.1/api/mgr/customers",
      params={
          'action'    :  'list_customer',
          'pagesize'  :  10,
          'pagenum'   :  1,
          'keywords'  :  '',
      })

printResponse(response)