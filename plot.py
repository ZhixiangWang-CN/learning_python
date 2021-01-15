import matplotlib.pyplot as plt
import numpy as np

# x = range(100)
# x = np.linspace(-3, 3, 50)
x = range(-10,10)
y1_list=[]
y2_list =[]
for i in x:
    y1 = 2 * i + 1
    y2 = i ** 2
    y1_list.append(y1)
    y2_list.append(y2)

plt.figure()
plt.plot(x, y2_list)
plt.plot(x, y1_list, color='red', linewidth=1.0, linestyle='--')
plt.show()