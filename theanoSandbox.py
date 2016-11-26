#http://deeplearning.net/software/theano/library/tensor/basic.html
#http://sinhrks.hatenablog.com/entry/2014/11/26/002818

#日本語解説はhttp://aidiary.hatenablog.com/entry/20150509/1431137590
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
f = function([x,y],z) #more flexible function definition, maybe preferrable consequnently
f(2,3) # function defined as f eats x,y,z,...
numpy.allclose(f(2,3),5)
z.eval({x : 2 , y : 3}) #z eats a dict type input through eval
if False:
    type(x.eval({x:1}))
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
T.printing.debugprint(z)
s = T.dscalar('s0')
t = r + s
u = function([p , q , s]  , [r , t])
u(1,2,3)
T.printing.debugprint(s)
x = T.dmatrix()
s = 1 / (1 + T.exp(-x))# expはcomponentwise
s.eval({x:[[1]]})
s.eval({x:[[1 , 2] , [3,4]]})
T.printing.debugprint(s)
from theano import pp#pretty print
pp(s)

f2 = T.function()

from theano import shared

state = shared(1 , name='state')
inc = T.iscalar('inc')
state.get_value()

accumulator = function(inputs = [inc] , outputs = [state] , updates=[(state  , state + inc)])#update happens after output evaluation
accumulator(109)