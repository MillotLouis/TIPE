from matplotlib import pyplot as plt
import numpy as np


x = np.arange(0,100,0.1)
y = 1/((x-90)**2)+1/((x)**2)

plt.plot(x,y,label="poids")
plt.legend()
plt.show()

