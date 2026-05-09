from matplotlib import pyplot as plt
import numpy as np


b = np.arange(0,100,0.1)
# fx = 1/((x-90)**2)+1/((x)**2)

# plt.plot(x,fx,label="f")

g = []
bi = 100
# s = 0.05 
# S = 10
# for i in range(len(b)):
#     if b[i] <= bi*s:
#         g.append(1)
#     else :
#         g.append(bi*s/(b[i]))

for i in range(len(b)):
    g.append((bi+0.01)/(0.01+b[i]))

plt.plot(b,g,label="g(b)")
plt.xlabel("batterie %")
plt.ylabel("g(b)")
# plt.axvline(x=5,color="orange",label="x=5")
plt.legend()
plt.show()
