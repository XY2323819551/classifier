"""
构建步骤：
（1）构建数据集，对应的X训练数据和Y标签，保存在tsv文件中 [ 数据X，对应类别，标签Y ]
（2）调用gensim包，为每一条数据都生成对应的vectors，另存为csv文件
（3）有了vectors和类别，就可以开始训练了

"""
