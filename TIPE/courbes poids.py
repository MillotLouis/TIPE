from matplotlib import pyplot as plt
import numpy as np


x = np.arange(0,100,0.1)
fx = 1/((x-90)**2)+1/((x)**2)

# plt.plot(x,fx,label="f")

seuil = 5 
norm = (x-seuil)/100
gx = 1/(0.1+norm)
for i in range(49):
    gx[i] = 20
gx /= 20

plt.plot(x,gx,label="g(x)")
plt.xlabel("batterie %")
plt.ylabel("g(x)")
# plt.axvline(x=5,color="orange",label="x=5")
plt.legend()
plt.show()
