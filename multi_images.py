import matplotlib.pyplot as plt
import numpy as np
X = np.arange(10)
y = X+1
z = 2*X**2+1

plt.figure()
plt.subplot(1,2,1)
plt.plot(X, y,'r--')
plt.subplot(1,2,2)
plt.plot(X, z)
plt.show()



