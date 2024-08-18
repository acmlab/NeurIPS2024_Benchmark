import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

def norm(vector):
    vector = np.array(vector)
    mean = np.mean(vector)
    std_dev = np.std(vector)
    normalized_vector = (vector - mean) / std_dev
    return normalized_vector

model_name = 'RNN'
direction = 'LR'
dataset = 'ABIDE'

X = torch.load('./feas_vis/'+model_name+'_'+dataset+'.pt')
y = np.loadtxt('./feas_vis/'+model_name+'_'+dataset+'.txt').astype(int)
X = np.array([np.array(x.detach().cpu().numpy()) for x in X])

model = LogisticRegression()
model.fit(X, y)
coefficients_abs = np.abs(model.coef_)
predict_X = model.predict(X)
coefficients = model.coef_
print(coefficients.shape, model.score(X, y))
np.savetxt('./attn_weights/'+model_name+'_'+dataset+'.txt', coefficients, fmt='%f')
