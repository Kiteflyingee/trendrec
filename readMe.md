# reference

1. PreprocessData.py  
   数据集预处理脚本  
delicous——sub2 文件格式  
uid    iid   timestamp  
内置方法  
```python
readData(filepath, split=',', train_ratio=0.8, sample_num=100000):
读取数据
split:列分隔符
train_ratio:训练集比例(前多少天为训练集)
sample_num:采样数量，我用来缩小数据集测试用的
```
主函数:执行切分操作

2. UserCF.py

CF推荐主要模块
```python  
UserCF类是主类，用于协同过滤算法推荐
get_item_score：用于获得item的最终得分
get_item_degree_distribute：获得训练集中item的度分布信息（废弃）
degree_item_map：获得训练集中item的度信息
accuracy：准确率指标(这次实验不用)
```

3. utils.py  
   一些工具方法



实验思路梳理  
1.在原先item推荐得分基础上，使得每个user贡献的得分*user的度  [x]  
2.筛选出训练集中度为1-10的集合（各自）  
3.以度为1的item为例 找出这些item  
   - 并找这些item在测试集里面的degree  
   - 对这些item按degree进行排序   

4.查询相关性评估方法
