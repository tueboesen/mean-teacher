from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os
from tabulate import tabulate

# from pandas import DataFrame
import pandas
# from collections import defaultdict

# 'load_pretrain': ".\\results\\cifar10_test\\2019-10-04_09_14_27\\100_10\\transient\\pretrain.120.ckpt",

# path = "D:\\Dropbox (CGI)\\Unsupervised\\mean-teacher\\pytorch\\results\\cifar10_test\\2019-10-08_08_56_57\\1000_10\\validation.msgpack"
path = "D:\\Dropbox (CGI)\\Unsupervised\\mean-teacher\\pytorch\\results\\cifar10_test\\2019-10-04_20_17_45\\1000_10\\validation.msgpack"
a=pandas.read_msgpack(path)
print(tabulate(a, headers='keys', tablefmt='psql'))

# a.to_csv('foo.txt', index=False, float_format='%g')
# df.to_csv('foo.txt', sep=chr(1))
print("did it work?")


# df = pd.DataFrame({'col_two' : [0.0001, 1e-005 , 1e-006, 1e-007],
#                    'column_3' : ['ABCD', 'ABCD', 'long string', 'ABCD']})
