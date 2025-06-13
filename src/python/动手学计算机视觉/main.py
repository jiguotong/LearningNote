import numpy as np
def add():
    A = np.array([[100,200],[300,400]], dtype=np.int32)
    B = np.array([[100,200],[300,400]], dtype=np.int32)
    C = A + B
    return C
    
C = np.ndarray((2,2),dtype=np.float32)      # 此时C.dtype为np.float32, id(C)=1894631025520
C = add()                                   # 此时C.dtype为np.int32, id(C)=1894631025040，此地址与函数内C=A+B的地址一致

def multiply(D):
    A = np.array([[100,200],[300,400]])
    B = np.array([[100,200],[300,400]])
    D = A * B

D = np.ndarray((2,2),dtype=np.float32)      # 此时D.dtype为np.float32, id(D)=1895112157200
multiply(D)                             # 此时D.dtype为np.int32, id(D)=1895496818192
pass