import numpy as np

import matplotlib.pyplot as plt
print(np.arange(0,10))
print([5]*10)

plt.fill_between(np.arange(0,11), [0]*11, [10]*11)

plt.plot([0,10,10,0,0],[0,0,100,100,0], color='black')
        
plt.show()