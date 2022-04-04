# -*- coding: utf-8 -*-
# @Time : 2021/10/11 11:06
# @Author : hhq
# @File : class_try.py
class house_item:
    k=3
    def __init__(self,name,area):

        self.name=name
        self.area=area
        # self.k = 3
    def __str__(self):
        return "%s,\n面积是%.2f" % (self.name,self.area)


class house:
    def __init__(self,house_type,house_area):
        self.type=house_type
        self.area=house_area
        self.free_area=house_area
        self.itemlist=[]
        self.house_item = house_item
    def __str__(self):
        return ("房子类型：%s\n房子面积：%.2f\n房子剩余面积：%.2f\n房子家具列表：%s\n"
                %(self.type,self.area,self.free_area,self.itemlist))

    def additem(self,item):
         print("要添加的家具是：%s"%item)
         if item.area >self.free_area:
             return "家具面积大，不能添加"
         self.free_area -=item.area
         self.itemlist.append(item.name)
    def im(self):
        self.k = self.house_item.k
        print(self.k)

bed=house_item("床",4)
print(bed)
myhome=house("两室一厅",60)
myhome.additem(bed)
# myhome.im()
print(myhome)