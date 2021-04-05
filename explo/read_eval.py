from numpy import load

import numpy as np
target = []

#norm values the model has been trained on
mean = -1.525913953781128
std =  2.279365062713623

mean = -5.4565  # -403.4768
std =  3.7140  # 244.4944



path_to_read = '/Users/rimassouel/Documents/ocp/results/is2re_predictions_0.npz'
data = load(path_to_read,allow_pickle=True)
lst = data.files


pred = data['energy']
targ = []
print(pred)

for batch in data['target'] :
    targ.extend(batch.tolist())
targ_scaled = []
print(targ)
for elem in targ :
    #targ_scaled.append((elem - mean)/std)
    targ_scaled.append(elem)

print(targ_scaled)
MAE = np.mean(np.abs(targ_scaled - pred))
MSE = np.mean((targ_scaled - pred)**2)

print('MSE : {}ev , MAE {}ev'.format(MSE, MAE))