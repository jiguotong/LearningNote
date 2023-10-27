import requests
import threading
import time


urls = [
    f'https://www.cnblogs.com/#p{page}'
    for page in range(1, 100 + 1)
]


def craw(url):
    r = requests.get(url)
    print(url, len(r.text))


# 单线程
def single_thread():
    print("single_thread begin")
    for url in urls:
        craw(url)
    print("single_thread end")
#
def multi_thread():
    print("multi_thread begin")
    threads = []
    for url in urls:    # urls中50个线程
        # 向这个list中添加threading.Thread的实例化对象
        threads.append(
            threading.Thread(target=craw,args=(url,))   # 这样写传进去的是元组
        )

    for thread in threads:
        thread.start() # 50个线程启动

    for thread in threads:
        thread.join() # 等待结束
    print("multi_thread end")


if __name__ == "__main__":
    start = time.time()
    single_thread()
    end = time.time()
    print("single thread cost:", end - start, "seconds")

    start = time.time()
    multi_thread()
    end = time.time()
    print("multi thread cost:", end - start, "seconds")