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
#If you provide no argument, the symbol will be unnamed. Names are not required, but they can help debugging.
x = T.dscalar('x')
y = T.dscalar('y')
z = x+ y
f = function([x,y],z)
f(2,3)
numpy.allclose(f(2,3),5)
z.eval({x : 2 , y : 3})
if False:
    tmp = {x : 2 , y : 3}
    type(tmp)
    tmp[x]
    z.eval(tmp)

p = T.dscalar('p0')
q = T.dscalar('q0')
r = p + q
r.eval({p:1 , q :1})
r.eval({p0:1 , q0 :1}) #error
r.eval({"p0":1 , "q0" :1}) #error
p = T.dscalar()
q = T.dscalar()
r = p + 2 * q
tmp = function([p , q] , r)
tmp(1,2)
r.eval({p:1 , q :1})

x = T.dmatrix()
s = 1 / (1 + T.exp(-x))# expはcomponentwise
s.eval({x:[[1]]})
s.eval({x:[[1 , 2] , [3,4]]})