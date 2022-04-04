# -*- coding: utf-8 -*-
# @Time : 2021/10/31 19:53
# @Author : hhq
# @File : Iterator_class.py
# 1,1,2,3,5,...
class Fib():
    def __init__(self):
        self.a, self.b = 0, 1
    def __iter__(self):
        return self
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        return self.a

fib = Fib()
for i in fib:
    if i > 10:
        # print(i)  # 13
        break
    print(i)  # 1,1,2,3,5,8