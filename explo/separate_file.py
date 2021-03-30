import pickle
from pymatgen.core.composition import Composition
import numpy as np
LIST = []
with open('all_list.txt', 'r') as file :

    for line in file.readlines() :
        if 'random' in line :
            elems = line.split('random')

            f = ('random'+elems[-1]).replace('\n', '')

            traj = f.split('.')[0]
            LIST.append(traj)

dic_data = pickle.load(open('/Users/rimassouel/Documents/ocp/explo/oc20_data_mapping.pkl','rb'))


#print(LIST)

OH_based = []
C1_based = []

OH_based_ad = []
C1_based_ad = []

REST_ad_noN = []

to_extract = []
for id in LIST :

    ads = dic_data[id]['ads_symbols']


    ads = ads.replace('*', '')
    comp = Composition(ads)
    formula = (comp.formula).split(' ')
    formula_no_digit = formula.copy()
    for j in range(len(formula)):
        formula_no_digit[j] = ''.join([i for i in formula[j] if not i.isdigit()])

    reduced = result = ''.join([i for i in comp.reduced_formula if not i.isdigit()])



    if 'N' not in formula_no_digit :
        if 'C' not in formula_no_digit :
            OH_based.append(id)
            OH_based_ad.append(ads)
        elif 'C2' not in formula  :
            C1_based.append(id)
            C1_based_ad.append(ads)





uniques_oh, count_oh = np.unique(OH_based_ad, return_counts=True)
uniques_c1, count_c1 = np.unique(C1_based_ad, return_counts=True)

to_extract.extend(OH_based)
to_extract.extend(C1_based)
uniques_res, count_rest = np.unique(REST_ad_noN, return_counts=True)

print(uniques_oh)
print(uniques_c1)


print(len(to_extract))
print(to_extract[:5])
with open('to_extract.txt', 'w') as f :
    for file in to_extract :
        file = 'is2res_train_trajectories/is2res_train_trajectories/'+file+'.extxyz.xz\n'
        f.write(file)









