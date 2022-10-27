import numpy as np
import matplotlib.pyplot as plt
#y=x+6x
rate = 0.01
#x_train = np.array([   1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#y_train = np.array([7,14,21,28,35,42,49,56,63,70,77,84,91,98,105,112,119,126,133,140])
x_train = np.array([   1,2,3,4,5,6,7,8,9,10,11])
y_train = np.array([7,14,21,28,35,42,49,56,63,70])
x_test  = np.array([    11,    12,    13,    14,   15,    16    ])

a = np.random.normal()
b = np.random.normal()


def h(x):
    return a*x+b*x

for i in range(90000):
    sum_a=0
    sum_b=0

    for x, y in zip(x_train, y_train):
        sum_a = sum_a + rate*(y-h(x))*x
        sum_b = sum_b + rate*(y-h(x))*x

    a = a + sum_a
    b = b + sum_b

    plt.plot([h(xi) for xi in x_test])

print(a)
print(b)


result=[h(xi) for xi in x_train]
print(result)

result=[h(xi) for xi in x_test]
print(result)

plt.show()