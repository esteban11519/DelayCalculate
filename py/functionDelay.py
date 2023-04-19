import numpy as np


# This was building from instructions cycles delay
def auxFunDelay(var, k):
    if k == 1:
        return 3*var[0]+6

    return (auxFunDelay(var, k-1)-2*k+3)*var[k-1]+2*k+4


def funDelay(var):
    '''
    var=[A, B, C .. ] from equations to delay
    return : Tbus or number os cicles of bus
    '''
    return auxFunDelay(var, var.size)


# var = np.ones(2, dtype=int)*256
var = np.array((2, 4, 3), dtype=int)
print(funDelay(var))
