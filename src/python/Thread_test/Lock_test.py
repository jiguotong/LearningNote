import threading
import time


# 实例化可重入锁类
lock = threading.RLock()
x = 0


class my_hreads(threading.Thread):
    """
    使用RLock实现线程同步，实现对临界资源的异步访问
    """
    def run(self):
        global x
        lock.acquire()      # 在操作变量x之前锁定资源
        for i in range(5):
            x += 10
        time.sleep(0.05)
        print(x)
        lock.release()      # 释放锁资源


if __name__ == '__main__':
    thrs = []
    for item in range(10):
        thrs.append(my_hreads())

    for item in thrs:
        item.start()

    pass

