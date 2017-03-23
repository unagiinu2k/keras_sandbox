#https://pymotw.com/3/itertools/index.html
from itertools import *
for i in chain([1,2,3]):
    print(i)


for i in chain([1,2,3] , ['a' , 'b']):
    print(i)

for i in zip([1,2,3] , ['a' , 'b' , 'cx']):
    print(i)

for i in zip([1,2,3] , ['a' , 'b']):
    print(i)

for i in zip_longest([1,2,3] , ['a' , 'b']):
    print(i)

#http://kk6.hateblo.jp/entry/20110521/1305984781

for i in "ABC": #"ABC" is iterator???
    print(i)

for k, g in groupby(sorted("AAAABBCCCDDDEEEFFAAACC"), key=list):
    #print(k) k is the group key(?)
    #print(g) g is the iterator within group
    print(k, len([i for i in g]), end=', ')