import threading

# 谁调用的join，谁就有一个命令当前线程阻塞的令牌，当前线程执行到这个语句的时候，就必须要等这个拥有令牌的线程结束之后才能继续执行
def print_num(num, thread_name=None, thr=None):
    for i in range(num):
        if i!=num/2:
            print(thread_name, '---->', i)
        else:
            if thr!=None:
                thr.join()
            else:
                print(thread_name, '---->', i)
    print(thread_name,' print done!')


class my_print_num_threads(threading.Thread):
    def __init__(self, num, thread_name=None, other_thread=None):
        super().__init__()
        self.num = num
        self.thread_name = thread_name
        self.other_thread = other_thread

    def run(self):
        for i in range(self.num):
            if i != self.num / 2:
                print(self.thread_name, '---->', i)
            else:
                if self.other_thread != None:
                    self.other_thread.join()
                else:
                    print(self.thread_name, '---->', i)
        print(self.thread_name, ' print done!')


if __name__ == '__main__':
    # method 1  直接在线程中运行函数
    thread_1 = threading.Thread(target=print_num, args=(100,'thread_1'))
    thread_2 = threading.Thread(target=print_num, args=(100,'thread_2',thread_1))

    thread_1.start()
    thread_2.start()

    ## 仅当thread_1和thread_2均执行完毕时，才运行method 2中的内容
    thread_1.join()
    thread_2.join()

    # method 2  通过继承类threading.Thread来创建线程，代码块写在类的run方法里面
    thread_1 = my_print_num_threads(100, 'thread_1', None)
    thread_2 = my_print_num_threads(100, 'thread_2', thread_1)
    thread_1.start()
    thread_2.start()

    ## 仅当thread_1和thread_2均执行完毕时，才运行done中的内容
    thread_1.join()
    thread_2.join()

    # done
    print('test done!')
    pass

