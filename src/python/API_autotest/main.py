import requests, json
import demo
import sys


for path in sys.path:
    print(path)

demo.test()

payload = {'key1': '加工', 'key2': '各个'}


# r = requests.get("http://www.baidu.com")
r = requests.post("http://httpbin.org/post",
                  data=payload)

print('---------------------------------------------')
print(r.status_code)
print('---------------------------------------------')
print(r.text)
print('---------------------------------------------')
# 要获取响应的消息体的文本内容，直接通过response对象 的 text 属性即可获取
# requests 会根据响应消息头（比如 Content-Type）对编码格式做推测，从而将消息体中的字符串解码为字符串
print(r.encoding)
print('---------------------------------------------')
print(r.text)
print('---------------------------------------------')
# 但是可能会预测不准，因此可以指定编码方式
r.encoding = 'utf8'
print(r.text)
print('---------------------------------------------')


# 如果我们要直接获取消息体中的字节串内容，可以使用 content 属性
print(r.content.decode('utf8'))

print('---------------------------------------------')
print(json.loads(r.content.decode('utf8')))
print(type(json.loads(r.content.decode('utf8'))))

print('---------------------------------------------')
print(r.json())
