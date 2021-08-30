
# defining the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
# No of data points used
N = 500
  
# normal distribution
data = np.random.randn(N)
  
# sort the data in ascending order
x = np.sort(data)
  
# get the cdf values of y
y = np.arange(N) / float(N)
  
# plotting
plt.xlabel('x-axis')
plt.ylabel('y-axis')
  
plt.title('CDF using sorting the data')
  
plt.plot(x, y, marker='o')
plt.show()

print(float(N))