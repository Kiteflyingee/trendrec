#coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''
''


class UserCF:
    '''
    基于用户的协同过滤
    '''
    def __init__(self):
#         记录用户相似性的字典，每到用的时候才计算 
        self.sim_dict = {}