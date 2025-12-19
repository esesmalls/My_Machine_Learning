import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
#%matplotlib inline


##操作数（Value）、算子、计算图和反向传播实现
class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):#fallback string representation
    return f"Value(data={self.data})"
  
  def __add__(self, other):#overload + operator
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():#闭包
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other): #overload * operator
    other = other if  isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def __rmul__(self, other):  #right-side multiplication
    return self * other
  
  def __pow__(self, other): #overload ** operator
    out = Value(self.data**other, (self, ), f'**{other}')
    
    def _backward():
      self.grad += other * (self.data**(other -1)) * out.grad
    out._backward = _backward
    
    return out
  def __neg__(self):  #overload unary - operator
    return self * Value(-1.0)
  
  def __sub__(self, other):   #overload - operator
    return self + (-other)
  
  def __truediv__(self, other):   #overload / operator
    return self * other**-1
  
  def exp(self):    #exponential function   
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def _backward(): 
      self.grad += out.data * out.grad
    out._backward = _backward
    
    return out
  
  def log(self):  #natural logarithm
    x = self.data
    out = Value(math.log(x), (self, ), 'log')
    
    def _backward():
      self.grad += (1/x) * out.grad
    out._backward = _backward
    
    return out

  def tanh(self):   #hyperbolic tangent function
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def backward(self): #backpropagation
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


##计算图的可视化

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)
  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

##多层感知机（MLP）实现
##单个神经元实现
class Neuron:
    def __init__(self, nin, layer_idx=1, neuron_idx=1):
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        # 将索引数字转为下标字符，便于在图中以下标形式展示
        digits = "₀₁₂₃₄₅₆₇₈₉"
        def sub(n):
            return "".join(digits[int(ch)] for ch in str(n))
        # 使用下标风格的参数标签 w_{层,神经元,输入} 和 b_{层,神经元}
        self.w = [Value(np.random.uniform(-1, 1), label=f"w{sub(layer_idx)}{sub(neuron_idx)}{sub(i+1)}") for i in range(nin)]#[]用于收集成为列表，作为神经元输入的权重
        self.b = Value(np.random.uniform(-1, 1), label=f"b{sub(layer_idx)}{sub(neuron_idx)}")#偏置用于控制神经元整体的触发频率
        
    def __call__(self, x):
        #n=n(x)=n.__call__(x)=w*x+b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)#self.b作为sum的初始值,可以增加计算效率
        #wi*xi for wi, xi in zip(self.w, x)作为生成器表达式，在python中必须用括号包裹
        out = act.tanh()
        return out
    
    def parameters(self):
        #收集Neuron参数，输出向量格式
        return self.w + [self.b]
    
##单层神经网络实现
class Layer:
    def __init__(self, nin, nout, layer_idx=1):
        #初始化层内的每一个独立神经元，根据Neuron的定义，初始参数都是随机的
        self.layer_idx = layer_idx
        self.neurons = [Neuron(nin, layer_idx=layer_idx, neuron_idx=i+1) for i in range(nout)]

    def __call__(self, x):
        #调用层时，将输入传递给层内的每个神经元（全连接），收集它们的输出
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

##多层感知机实现   
class MLP:
    def __init__(self, nin, nouts):
        ##这里的nouts是所有Layer中的nout构成的向量。nin是初始层接受的输入向量维度
        ##后续层接受的输入层维度由上一层的nout决定，因此设计sz和Layer(sz[i],sz[i+1])
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1], layer_idx=i+1) for i in range(len(nouts))]
    def __call__(self,x):
        #输入向量在不同层迭代更新
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for Layer in self.layers for p in Layer.parameters()]

##示例：创建一个MLP并进行前向和反向传播
T = MLP(3,[4,4,1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]##这是期望输出，看得出来是一个二分类问题
ypred = [T(x) for x in xs]

for k in range(20):

    for p in T.parameters():
        p.grad = 0.0 #重置梯度，否则会累加
    ypred = [T(x) for x in xs]
    loss = sum(((yout-ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
    #在对T的参数梯度重置后，无需再对loss进行梯度重置，循环变量
    #ygt，yout的grad继承ys和ypred，而且不会累加，ys是常量0梯度，ypred继承T参数梯度
    #另外，被减的ygt是ys常量，画图过程中不会产生多余节点，如果反过来会吗？
    loss.backward()
    
    for p in T.parameters():
        p.data += -0.05 * p.grad

    print(k,loss.data)
draw_dot(loss).render('mlp_graph', view=True)