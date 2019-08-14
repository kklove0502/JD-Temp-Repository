import numpy as np

def computeCorrelation(x,y):
    """
    计算功能函数，计算简单线性回归相关系数的函数
    :param x,y: series，计算相关系数的两个序列
    :return: r：float，相关系数
    """
    import math
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0,len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar**2
        varY += difYYbar**2
    SST = math.sqrt(varX * varY)
    return SSR/SST