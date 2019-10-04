import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = np.arange(9).reshape(3, 3)
print(data)
data = StandardScaler().fit_transform(X=data)
print(data)