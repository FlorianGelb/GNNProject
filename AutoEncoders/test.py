import shap
import numpy as np

def f(x):
  ## x is a 3D vector
  return x[:, 0] + 2*x[:,1] - 3*x[:,2]

data = np.random.rand(10,3)
explainer = shap.KernelExplainer(f, data, algorithm='permutation')
shap_values = explainer(data)
print(shap_values)