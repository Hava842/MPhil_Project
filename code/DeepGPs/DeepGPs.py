import numpy as np
import matplotlib.pyplot as plt
from GaussianProcess import GaussianProcess

gp = GaussianProcess()
X = np.random.uniform(0, 1, (10, 1))
y = (np.sin((X[:, 0])*10) - X[:,0]*3).reshape(-1, 1)
gp.fit(X, y, train=True, iterations=10000)
X_ = np.linspace(0, 1, 1000).reshape(-1, 1)
y_, var_ = gp.predict(X_)
std_ = np.sqrt(np.clip(var_, 0, np.inf)).reshape(-1, 1)

plt.scatter(X, y)
plt.plot(X_, y_, c='r')
plt.plot(X_, y_+2*std_, c='g')
plt.plot(X_, y_-2*std_, c='g')
plt.show()