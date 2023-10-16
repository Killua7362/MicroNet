import numpy as np

from micro.tensor import Tensor

class Parameters(Tensor):
    def __init__(self,*shapes) -> None:
        data = np.random.randn(*shapes)
        super().__init__(data,requires_grad=True)