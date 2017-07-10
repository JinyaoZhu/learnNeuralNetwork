from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
import random

style.use('ggplot')

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return (np.array(xs,dtype=np.float64), 
            np.array(ys,dtype=np.float64))


xs,ys = create_dataset(40,10,2,correlation='pos')

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys))-mean(xs*ys))/
        ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m,b

m,b = best_fit_slope(xs,ys)

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)*(ys_line - ys_orig))

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for _ in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

regression_line = [(m*x)+b for x in xs]

r_squared = coefficient_of_determination(ys,regression_line)

predict_x = xs[-1]+1
predict_y = (m*predict_x)+b

plt.scatter(xs,ys,color='#003F72',label='data')
plt.scatter(predict_x,predict_y,label='predict')
plt.plot(xs,regression_line,label='regression line')
plt.legend(loc=4)
plt.show()

print(r_squared)