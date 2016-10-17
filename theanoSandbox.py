#http://deeplearning.net/software/theano/library/tensor/basic.html
#http://sinhrks.hatenablog.com/entry/2014/11/26/002818
import theano.tensor as T
x = T.fmatrix()
x.shape
x = T.iscalar('myvar')

import numpy
y=numpy.asarray([1,2])
y2=numpy.array([1,2])

from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x+ y
f = function([x,y],z)
f(2,3)