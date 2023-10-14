class Tensor:
    def __init__(self,data,node=(),op=''):
        self.data = data
        self._prev = set(node)
        self._op = op
        
    def __repr__(self):
        return f'Tensor(data={self.data})'
    
    def __add__(self,other):
        return Tensor(self.data + other.data,(self,other),'+')
    
    def __mul__(self,other):
        return Tensor(self.data * other.data,(self,other),'*')
    