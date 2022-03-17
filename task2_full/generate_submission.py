import yaml
import pandas as pd
import numpy as np

from pathlib import Path
from baseline import prepare_model
from baseline import read_pymatgen_dict
from baseline import prepare_dataset # really needed?
from sklearn.ensemble import GradientBoostingRegressor



def main(config):
    public = prepare_dataset(config['datapath'], test_mode = 'public')
    private = prepare_dataset(config['test_datapath'], test_mode = 'private') #изменить на функцию Вовы! где без строчки про targets
    df_train = public[0]
    df_test = public[1]    # это не сам тест, а валидация!
    pr_test = private # сам тест
    
    # number of atoms in the lattice
    def atoms_num(i):
        return len(i.as_dict()['sites'])
    
    for k in [df_train, df_test, pr_test]: ### I addd sth here!!!
        k['atoms_num'] = k.iloc[:, 0].map(atoms_num)
        
        
    def atoms_per_lattice(i):
        sites = i.as_dict()['sites']
        return set([sites[j]['label'] for j in range(len(sites))])
    
    sets_atoms_per_lattice = df_train.iloc[:, 0].map(atoms_per_lattice) # только train
    
    unique_atoms_train = set() 
    for i in sets_atoms_per_lattice:
        unique_atoms_train = unique_atoms_train.union(i)
    
    # only four atoms!!!
    unique_atoms_train = list(unique_atoms_train)
    
    from collections import Counter
    
    # обработка Train   
    count_atoms_per_lattice = dict()
    for i in unique_atoms_train:
        count_atoms_per_lattice[i] = []
        
        
    for i in df_train.iloc[:, 0]:
        sites = i.as_dict()['sites']
        count_atoms = Counter([sites[j]['label'] for j in range(len(sites))])
        for j in unique_atoms_train:
            if j in count_atoms.keys():
                count_atoms_per_lattice[j].append(count_atoms[j])
            else:
                count_atoms_per_lattice[j].append(0)
            
    df_train['Mo_count'] = count_atoms_per_lattice['Mo'] 
    df_train['S_count'] = count_atoms_per_lattice['S']
    df_train['Se_count'] = count_atoms_per_lattice['Se']
    df_train['W_count'] = count_atoms_per_lattice['W']
    
    # Обработка Test
    count_atoms_per_lattice_test = dict()
    for i in unique_atoms_train: ## why train? same for test and pr_test?
        count_atoms_per_lattice_test[i] = []
    
    for i in df_test.iloc[:, 0]:
        sites = i.as_dict()['sites']
        count_atoms = Counter([sites[j]['label'] for j in range(len(sites))])
        for j in unique_atoms_train:  ## why train? same for test and pr_test?
            if j in count_atoms.keys():
                count_atoms_per_lattice_test[j].append(count_atoms[j])
            else:
                count_atoms_per_lattice_test[j].append(0)
    
    df_test['Mo_count'] = count_atoms_per_lattice_test['Mo'] 
    df_test['S_count'] = count_atoms_per_lattice_test['S']
    df_test['Se_count'] = count_atoms_per_lattice_test['Se']
    df_test['W_count'] = count_atoms_per_lattice_test['W']
    
    ##### Обработка Private
    count_atoms_per_lattice_pr_test = dict()
    for i in unique_atoms_train:
        count_atoms_per_lattice_pr_test[i] = []
    
    for i in pr_test.iloc[:, 0]:
        sites = i.as_dict()['sites']
        count_atoms = Counter([sites[j]['label'] for j in range(len(sites))])
        for j in unique_atoms_train:
            if j in count_atoms.keys():
                count_atoms_per_lattice_pr_test[j].append(count_atoms[j])
            else:
                count_atoms_per_lattice_pr_test[j].append(0)
    
    pr_test['Mo_count'] = count_atoms_per_lattice_pr_test['Mo'] # 
    pr_test['S_count'] = count_atoms_per_lattice_pr_test['S']
    pr_test['Se_count'] = count_atoms_per_lattice_pr_test['Se']
    pr_test['W_count'] = count_atoms_per_lattice_pr_test['W']
    #####
    
    ## choosing only needed columns and creating binary variables
    df_train1 = df_train[['Mo_count', 'W_count', 'Se_count', 'S_count', 'targets']]
    df_train1 = pd.get_dummies(df_train1, drop_first = True, columns = ['Mo_count', 'W_count', 'Se_count', 'S_count'])
    
    df_test1 = df_test[['Mo_count', 'W_count', 'Se_count', 'S_count', 'targets']]
    df_test1 = pd.get_dummies(df_test1, drop_first = True, columns = ['Mo_count', 'W_count', 'Se_count', 'S_count'])   
    
    ### same for Private
    pr_test1 = pr_test[['Mo_count', 'W_count', 'Se_count', 'S_count', 'targets']]
    pr_test1 = pd.get_dummies(pr_test1, drop_first = True, columns = ['Mo_count', 'W_count', 'Se_count', 'S_count'])
    
    for i in df_train1.columns:
        if i not in df_test1.columns:
            df_test1[i] = 0
            ### adding
        if i not in pr_test1.columns:
            pr_test1[i] = 0
    
    
    df_test1.columns = df_train1.columns
    ### adding
    pr_test1.columns = df_train1.columns
    
    df_train1['atoms_count_group'] = 0
    for i, name in enumerate(df_train1.columns[1:7]):
        df_train1['atoms_count_group'] = df_train1['atoms_count_group'] + df_train1[name] * 10**(6-i)
    
    df_test1['atoms_count_group'] = 0
    for i, name in enumerate(df_test1.columns[1:7]):
        df_test1['atoms_count_group'] = df_test1['atoms_count_group'] + df_test1[name] * 10**(6-i)
    
    ### copy-paste for private
    
    pr_test1['atoms_count_group'] = 0
    for i, name in enumerate(pr_test1.columns[1:7]):
        pr_test1['atoms_count_group'] = pr_test1['atoms_count_group'] + pr_test1[name] * 10**(6-i)
        
        
        
    
    ################################ adding 10000 group creator
    # selecting only 10000 group
    df_train1 = df_train1.reset_index()
    df_test1 = df_test1.reset_index()
    pr_test1 = pr_test1.reset_index()
    
    df_train1_10000 = df_train1[df_train1['atoms_count_group'] == 10000].reset_index().copy()
    train_index_10000 = df_train1[df_train1['atoms_count_group'] == 10000].index
    
    df_test1_10000 = df_test1[df_test1['atoms_count_group'] == 10000].reset_index().copy()
    test_index_10000 = df_test1[df_test1['atoms_count_group'] == 10000].index
    
    df_private1_10000 = pr_test1[pr_test1['atoms_count_group'] == 10000].reset_index().copy()
    private_index_10000 = pr_test1[pr_test1['atoms_count_group'] == 10000].index
    # abc of Se atom
    Se_abc_10000_train = []
    for i in train_index_10000.to_list():                         
        sample = df_train.iloc[i, 0].as_dict()['sites']
        for j in sample:
            if j['label'] == 'Se':
                Se_abc_10000_train.append(j['abc'])
    
    
    # do the same for the test
    Se_abc_10000_test = []
    for i in test_index_10000.to_list():                          
        sample = df_test.iloc[i, 0].as_dict()['sites']
        for j in sample:
            if j['label'] == 'Se':
                Se_abc_10000_test.append(j['abc'])
    
    # do the same for the private
    Se_abc_10000_private = []
    for i in private_index_10000.to_list():                       
        sample = private.iloc[i, 0].as_dict()['sites']
        for j in sample:
            if j['label'] == 'Se':
                Se_abc_10000_private.append(j['abc'])
                
    # creating dataframe from that 
    df_Se_abc_10000_train = pd.DataFrame(Se_abc_10000_train)
    df_Se_abc_10000_train.columns = ['a', 'b', 'c']
    
    df_Se_abc_10000_test = pd.DataFrame(Se_abc_10000_test)
    df_Se_abc_10000_test.columns = ['a', 'b', 'c']
    
    df_Se_abc_10000_private = pd.DataFrame(Se_abc_10000_private)
    df_Se_abc_10000_private.columns = ['a', 'b', 'c']
    # adding to current df
    df_train1_10000['Se_a'] = round(df_Se_abc_10000_train['a'], 3)
    df_train1_10000['Se_b'] = round(df_Se_abc_10000_train['b'], 3)
    df_train1_10000['Se_c'] = round(df_Se_abc_10000_train['c'], 3)
    
    df_test1_10000['Se_a'] = round(df_Se_abc_10000_test['a'], 3)
    df_test1_10000['Se_b'] = round(df_Se_abc_10000_test['b'], 3)
    df_test1_10000['Se_c'] = round(df_Se_abc_10000_test['c'], 3)
    
    df_private1_10000['Se_a'] = round(df_Se_abc_10000_private['a'], 3)
    df_private1_10000['Se_b'] = round(df_Se_abc_10000_private['b'], 3)
    df_private1_10000['Se_c'] = round(df_Se_abc_10000_private['c'], 3)
    # creating all 64 possible coordinates for Molibden and each layer of S (ANY GROUP IS THE SAME!!!)
    # Molibden
    Mo_a_set = []
    Mo_b_set = []
    Mo_c_set = []
    # S, 1st and 2nd layer
    S_a_set = []
    S_b_set = []
    S_c_set = []
    
    for val, i in enumerate(df_train.iloc[0, 0].as_dict()['sites']):
        if val < 63:
            Mo_a_set.append(round(i['abc'][0], 3))
            Mo_b_set.append(round(i['abc'][1], 3))
            Mo_c_set.append(round(i['abc'][2], 3))
        # skipping Se atom
        elif val > 64:
            S_a_set.append(round(i['abc'][0], 3))
            S_b_set.append(round(i['abc'][1], 3))
            S_c_set.append(round(i['abc'][2], 3))
    
    #print('Molibden')
    Mo_a_set = sorted(list(set(Mo_a_set)))
    Mo_b_set = sorted(list(set(Mo_b_set)))
    Mo_c_set = sorted(list(set(Mo_c_set)))[0]
    #print(Mo_a_set)
    #print(Mo_b_set)
    #print(Mo_c_set)
    
    #print('S')
    S_a_set = sorted(list(set(S_a_set)))
    S_b_set = sorted(list(set(S_b_set)))
    S_c_set = sorted(list(set(S_c_set)))
    #print(S_a_set)
    #print(S_b_set)
    #print(S_c_set)
    # LOL, it's just vice versa a and b for Mo and S!!!
    # creating list of 64 places for Mo and 128 places for S in the correct order
    Mo_64_coordinates = []
    for i in Mo_a_set:
        for j in Mo_b_set:
            Mo_64_coordinates.append([i, j, Mo_c_set])
    
    S_128_coordinates = []
    for i in S_c_set:
        for j in S_a_set:
            for k in S_b_set:
                S_128_coordinates.append([j, k, i])
    # DETECTING MISSED ATOMS(USING ALGORITHM ABOVE) FOR ALL LATTICES IN THE GROUP 10000!!!!
    num_molibden_atoms = 63
    num_rubbish_atoms = 1
    num_S_atoms = 126
    
    Mo_missed = 64 - num_molibden_atoms
    S_missed = 128 - num_S_atoms
    
    missed_atoms_coordinates_group10000_train = []
    missed_atoms_coordinates_group10000_test = []
    missed_atoms_coordinates_group10000_private = []
    
    indices = [train_index_10000, test_index_10000, private_index_10000]
    coordinates_list = [missed_atoms_coordinates_group10000_train,
                        missed_atoms_coordinates_group10000_test,
                        missed_atoms_coordinates_group10000_private]
    datasets = [df_train, df_test, private]
    
    for ind in range(3):
        for j in indices[ind]:
            missed_atoms_coordinates_lattice = []
    
            Mo_atom_missed_counter = 0
            S_atom_missed_counter = 0
    
            lattice_sample = datasets[ind].iloc[j, 0].as_dict()['sites']
    
            for val, i in enumerate(lattice_sample):
                # first Mo atom
                if val == 0 and round(i['abc'][1], 3) != 0.083:
                    missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + Mo_64_coordinates[val]
                    Mo_atom_missed_counter += 1
                # other Mo atoms
                elif val > 0 and val < num_molibden_atoms:
                    diff_b_with_prev = round(i['abc'][1] - lattice_sample[val - 1]['abc'][1], 3)
                    if (diff_b_with_prev != 0.125) & (diff_b_with_prev != -0.875):
                        missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + Mo_64_coordinates[val]
                        Mo_atom_missed_counter += 1
    
                # first S atom
                elif val == num_molibden_atoms + num_rubbish_atoms:
                    if round(i['abc'][1], 3) != 0.042:
                        missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + S_128_coordinates[val - num_molibden_atoms - num_rubbish_atoms + S_atom_missed_counter]
                        S_atom_missed_counter += 1
    
                # other S atoms
                elif val > num_molibden_atoms + num_rubbish_atoms:
                    diff_b_with_prev = round(i['abc'][1] - lattice_sample[val - 1]['abc'][1], 3)
                    if (diff_b_with_prev != 0.125) & (diff_b_with_prev != -0.875):
                        missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + S_128_coordinates[val - num_molibden_atoms - num_rubbish_atoms + S_atom_missed_counter]
                        S_atom_missed_counter += 1
                        # Case if missed atoms are together!!! This is special case to be algorithmed below!!!!
                    # UPD: MISTAKE, --------------------------------------------------------------------------------------------------------> DON'T ADD 1 BELOW!!!! AS S_atom_missed_counter was updated!!!!
                        if (diff_b_with_prev != 0.25) & (diff_b_with_prev != -0.75):
                            missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + S_128_coordinates[val - num_molibden_atoms - num_rubbish_atoms + S_atom_missed_counter]
                            S_atom_missed_counter += 1
                            break
    
            # if missed atoms are on last positions, so the algorithm above does not catch them
            # Molibden
            if Mo_atom_missed_counter != Mo_missed:
                missed_atoms_coordinates_lattice = Mo_64_coordinates[63] + missed_atoms_coordinates_lattice
                Mo_atom_missed_counter += 1
                
            # S
            if S_atom_missed_counter - S_missed == -2:
                missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + S_128_coordinates[126]
                missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + S_128_coordinates[127]
                S_atom_missed_counter += 2
            elif S_atom_missed_counter - S_missed == -1:
                missed_atoms_coordinates_lattice = missed_atoms_coordinates_lattice + S_128_coordinates[127]
                S_atom_missed_counter += 1
    
            coordinates_list[ind].append(missed_atoms_coordinates_lattice)
    # df of missed atoms!!!
    df_missed_coord_10000_train = pd.DataFrame(missed_atoms_coordinates_group10000_train)
    df_missed_coord_10000_test = pd.DataFrame(missed_atoms_coordinates_group10000_test)
    df_missed_coord_10000_private = pd.DataFrame(missed_atoms_coordinates_group10000_private)
    main_datasets = [df_train1_10000, df_test1_10000, df_private1_10000]
    prelim_datasets = [df_missed_coord_10000_train, df_missed_coord_10000_test, df_missed_coord_10000_private]
    
    for i in range(3):
        main_datasets[i]['missed_Mo_a'] = prelim_datasets[i][0]
        main_datasets[i]['missed_Mo_b'] = prelim_datasets[i][1]
        main_datasets[i]['missed_Mo_c'] = prelim_datasets[i][2]
        main_datasets[i]['missed_S1_a'] = prelim_datasets[i][3]
        main_datasets[i]['missed_S1_b'] = prelim_datasets[i][4]
        main_datasets[i]['missed_S1_c'] = prelim_datasets[i][5]
        main_datasets[i]['missed_S2_a'] = prelim_datasets[i][6]
        main_datasets[i]['missed_S2_b'] = prelim_datasets[i][7]
        main_datasets[i]['missed_S2_c'] = prelim_datasets[i][8]
    cols_to_leave = ['level_0', 'index', 'targets', 'Se_a', 'Se_b', 'Se_c',
          'missed_Mo_a', 'missed_Mo_b', 'missed_Mo_c', 'missed_S1_a',
          'missed_S1_b', 'missed_S1_c', 'missed_S2_a', 'missed_S2_b',
          'missed_S2_c']
    
    train10000 = df_train1_10000[cols_to_leave]
    test10000 = df_test1_10000[cols_to_leave]
    private10000 = df_private1_10000[cols_to_leave]
    def substantial_impurity_detector(i):
        if round(i['Se_a'], 3) == i['missed_S1_a'] and round(i['Se_b'], 3) == i['missed_S1_b'] and round(i['Se_c'], 3) == i['missed_S1_c']:
            return 1
        elif round(i['Se_a'], 3) == i['missed_S2_a'] and round(i['Se_b'], 3) == i['missed_S2_b'] and round(i['Se_c'], 3) == i['missed_S2_c']:
            return 2
        else:
            return 0
    train10000['impurity_atom'] = train10000.apply(substantial_impurity_detector, axis = 1)
    test10000['impurity_atom'] = test10000.apply(substantial_impurity_detector, axis = 1)
    private10000['impurity_atom'] = private10000.apply(substantial_impurity_detector, axis = 1)
    train10000_new = train10000.copy()
    test10000_new = test10000.copy()
    private10000_new = private10000.copy()
    
    new_dfs = []
    
    for df in [train10000_new, test10000_new, private10000_new]:
        df['vacancyS_a'] = df.apply(lambda x: x['missed_S1_a'] if x['impurity_atom'] == 2 else x['missed_S2_a'], axis = 1)
        df['vacancyS_b'] = df.apply(lambda x: x['missed_S1_b'] if x['impurity_atom'] == 2 else x['missed_S2_b'], axis = 1)
        df['vacancyS_c'] = df.apply(lambda x: x['missed_S1_c'] if x['impurity_atom'] == 2 else x['missed_S2_c'], axis = 1)
    
        df['Se_a'] = np.round(df['Se_a'], 3)
        df['Se_b'] = np.round(df['Se_b'], 3)
        df['Se_c'] = np.round(df['Se_c'], 3)
    
        cols_to_leave = ['level_0', 'index', 'targets', 'Se_a', 'Se_b', 'Se_c',
              'missed_Mo_a', 'missed_Mo_b', 'missed_Mo_c', 'vacancyS_a', 'vacancyS_b', 'vacancyS_c']
    
        df = df[cols_to_leave].copy()
    
        new_col_names = ['level_0', 'index', 'targets', 'Se_impurityS_a', 'Se_impurityS_b', 'Se_impurityS_c',
              'vacancyMo_a', 'vacancyMo_b', 'vacancyMo_c', 'vacancyS_a', 'vacancyS_b', 'vacancyS_c']
        df.columns = new_col_names
    
        new_dfs.append(df)
    
    train10000_new = new_dfs[0]
    test10000_new = new_dfs[1]
    private10000_new = new_dfs[2]
    # FROM OBSERVATION OF THE COORDINATES!!!
    mean_a = 0.5
    mean_b = 0.5
    mean_c = 0.25
    
    for df in [train10000_new, test10000_new, private10000_new]:
        df['Se_impurityS_dist_to_center'] = np.sqrt((df['Se_impurityS_a']-mean_a)**2 + (df['Se_impurityS_b']-mean_b)**2 + (df['Se_impurityS_c']-mean_c)**2)
        df['vacancyMo_dist_to_center'] = np.sqrt((df['vacancyMo_a']-mean_a)**2 + (df['vacancyMo_b']-mean_b)**2 + (df['vacancyMo_c']-mean_c)**2)
        df['vacancyS_dist_to_center'] = np.sqrt((df['vacancyS_a']-mean_a)**2 + (df['vacancyS_b']-mean_b)**2 + (df['vacancyS_c']-mean_c)**2)
    train_df10000 = train10000_new.copy()
    valid_df10000 = test10000_new.copy()
    private_df10000 = private10000_new.copy()
    # another 3 features, All pairwise distances between the vacancy and impurity coordinates!!!
    dfs = [train_df10000, valid_df10000, private_df10000]
    for df in dfs:
        df['dist_Se_impurityS_vacancyMo'] = np.sqrt((df['Se_impurityS_a'] - df['vacancyMo_a'])**2\
                                                            + (df['Se_impurityS_b'] - df['vacancyMo_b'])**2\
                                                            + (df['Se_impurityS_c'] - df['vacancyMo_c'])**2)
        df['dist_Se_impurityS_vacancyS'] = np.sqrt((df['Se_impurityS_a'] - df['vacancyS_a'])**2\
                                                                + (df['Se_impurityS_b'] - df['vacancyS_b'])**2\
                                                                + (df['Se_impurityS_c'] - df['vacancyS_c'])**2)
        df['dist_vacancyMo_vacancyS'] = np.sqrt((df['vacancyMo_a'] - df['vacancyS_a'])**2\
                                                            + (df['vacancyMo_b'] - df['vacancyS_b'])**2\
                                                            + (df['vacancyMo_c'] - df['vacancyS_c'])**2)
    
    # euclidean sum of distances
    for df in dfs:
        df['eucl_sum_distance'] = np.sqrt(df['dist_Se_impurityS_vacancyMo']**2 +\
                                          df['dist_Se_impurityS_vacancyS']**2 +\
                                          df['dist_vacancyMo_vacancyS']**2)
        
    # another 3 features: distances from the origin
    for df in dfs:
        df['dist_Se_impurityS_origin'] = np.sqrt((df['Se_impurityS_a'])**2\
                                                              + (df['Se_impurityS_b'])**2\
                                                              + (df['Se_impurityS_c'])**2)
        df['dist_vacancyS_origin'] = np.sqrt((df['vacancyS_a'])**2\
                                                          + (df['vacancyS_b'])**2\
                                                          + (df['vacancyS_c'])**2)
        df['dist_vacancyMo_origin'] = np.sqrt((df['vacancyMo_a'])**2\
                                                          + (df['vacancyMo_b'])**2\
                                                          + (df['vacancyMo_c'])**2)
    def cos_creator1(i):
        dist2D = np.sqrt(i['dist_vacancyMo_vacancyS']**2 - (i['vacancyMo_c'] - i['vacancyS_c'])**2)
        if dist2D == 0:
            return 0
        # pivot is vacancyMo:
        elif i['vacancyMo_c'] < i['vacancyS_c']:
            return (i['vacancyS_a'] - i['vacancyMo_a']) / dist2D
        # pivot is vacancyS
        else:
            return (i['vacancyMo_a'] - i['vacancyS_a']) / dist2D
    
    def sin_creator1(i):
        dist2D = np.sqrt(i['dist_vacancyMo_vacancyS']**2 - (i['vacancyMo_c'] - i['vacancyS_c'])**2)
        if dist2D == 0:
            return 0
        # pivot is vacancyMo:
        elif i['vacancyMo_c'] < i['vacancyS_c']:
            return (i['vacancyS_b'] - i['vacancyMo_b']) / dist2D
        else:
            return (i['vacancyMo_b'] - i['vacancyS_b']) / dist2D
    
    
    def cos_creator2(i):
        dist2D = np.sqrt(i['dist_Se_impurityS_vacancyS']**2 - (i['Se_impurityS_c'] - i['vacancyS_c'])**2)
        if dist2D == 0:
            return 0
        # pivot is Se_impurityS:
        elif i['Se_impurityS_c'] < i['vacancyS_c']:
            return (i['vacancyS_a'] - i['Se_impurityS_a']) / dist2D
        # pivot is vacancyS
        else:
            return (i['Se_impurityS_a'] - i['vacancyS_a']) / dist2D
    
    def sin_creator2(i):
        dist2D = np.sqrt(i['dist_Se_impurityS_vacancyS']**2 - (i['Se_impurityS_c'] - i['vacancyS_c'])**2)
        if dist2D == 0:
            return 0
        # pivot is Se_impurityS:
        elif i['Se_impurityS_c'] < i['vacancyS_c']:
            return (i['vacancyS_b'] - i['Se_impurityS_b']) / dist2D
        else:
            return (i['Se_impurityS_b'] - i['vacancyS_b']) / dist2D
    
    
    def cos_creator3(i):
        dist2D = np.sqrt(i['dist_Se_impurityS_vacancyMo']**2 - (i['vacancyMo_c'] - i['Se_impurityS_c'])**2)
        if dist2D == 0:
            return 0
        # pivot is vacancyMo:
        elif i['vacancyMo_c'] < i['Se_impurityS_c']:
            return (i['Se_impurityS_a'] - i['vacancyMo_a']) / dist2D
        # pivot is Se_impurityS
        else:
            return (i['vacancyMo_a'] - i['Se_impurityS_a']) / dist2D
    
    def sin_creator3(i):
        dist2D = np.sqrt(i['dist_Se_impurityS_vacancyMo']**2 - (i['vacancyMo_c'] - i['Se_impurityS_c'])**2)
        if dist2D == 0:
            return 0
        # pivot is vacancyMo:
        elif i['vacancyMo_c'] < i['Se_impurityS_c']:
            return (i['Se_impurityS_b'] - i['vacancyMo_b']) / dist2D
        else:
            return (i['vacancyMo_b'] - i['Se_impurityS_b']) / dist2D
    # applying tricky angle!!!
    dfs = [train_df10000, valid_df10000, private_df10000]
    for df in dfs:
        df['sin_tricky1'] = df.apply(sin_creator1, axis = 1)
        df['cos_tricky1'] = df.apply(cos_creator1, axis = 1)
        df['tan_tricky1'] = df['sin_tricky1'] / (df['cos_tricky1'] + 0.00001)
        df['ctg_tricky1'] = df['cos_tricky1'] / (df['sin_tricky1'] + 0.00001)
    
        df['sin_tricky2'] = df.apply(sin_creator2, axis = 1)
        df['cos_tricky2'] = df.apply(cos_creator2, axis = 1)
        df['tan_tricky2'] = df['sin_tricky2'] / (df['cos_tricky2'] + 0.00001)
        df['ctg_tricky2'] = df['cos_tricky2'] / (df['sin_tricky2'] + 0.00001)
    
        df['sin_tricky3'] = df.apply(sin_creator3, axis = 1)
        df['cos_tricky3'] = df.apply(cos_creator3, axis = 1)
        df['tan_tricky3'] = df['sin_tricky3'] / (df['cos_tricky3'] + 0.00001)
        df['ctg_tricky3'] = df['cos_tricky3'] / (df['sin_tricky3'] + 0.00001)
    # another 2 features, binary variables, depending on the C coordinates of the atoms on S places!!!
    dfs = [train_df10000, valid_df10000, private_df10000]
    for df in dfs:
        df['C_atomsS_same'] = (df['vacancyS_c'] == df['Se_impurityS_c']).astype('int')
        df['vacancyS_c_0_355'] = (df['vacancyS_c'] == 0.355).astype('int')
    
   
    
    ################################# ending 10000 group creator: train_df10000, valid_df10000, private_df10000 as an output!!!!!!!!!!!!!!!!!!!!!!!!     
    
    
    # prep feautres and run AutoML
    train_df10000.drop(['vacancyMo_c'],axis=1, inplace=True)
    valid_df10000.drop(['vacancyMo_c'],axis=1, inplace=True)
    private_df10000.drop(['vacancyMo_c'],axis=1, inplace=True)
  
    X_train = train_df10000.drop(['targets', 'level_0', 'index'], axis=1)
    y_train = train_df10000.targets
    X_test = valid_df10000.drop(['targets', 'level_0', 'index'], axis=1)
    y_test = valid_df10000.targets
    X_priv = private_df10000.drop(['targets', 'level_0', 'index'], axis=1)
    # no targets for priv
    y_zero = private_df10000.targets
    
    
    
    gb = GradientBoostingRegressor(loss='huber', criterion='squared_error', n_estimators = 1000, max_depth = 1, learning_rate = 1)
    gb.fit(X_train, y_train)
    priv_pred_gr10000 = gb.predict(X_priv)
    df_priv_pred_gr10000 = pd.DataFrame(data={'predictions': priv_pred_gr10000}, index=private_df10000['index'])
    
    
        
    df_train1 = df_train1.set_index('index')
    df_test1 = df_test1.set_index('index')
    pr_test1 = pr_test1.set_index('index') 
    
    #y_pred_train_easy = df_train1.groupby('atoms_count_group')['targets'].transform('median')
    #y_pred_test_easy = df_test1.groupby('atoms_count_group')['targets'].transform('median')
    df_pub = pd.concat([df_train1, df_test1], axis=0)
    
    # add-in: replace median with mode
    # 0.02 LENGTH WINDOW!!!!!!
    for d in [df_pub, df_train1, df_test1]:
        # MAKING UNIVERSAL ROUND, SO THERE'LL BE ROUND NOT ONLY TO CLOSEST 0.01, 0.1, 0.001 ETC, BUT FOR 0.005, 0.0025, ETC!!!!!!
        d['targets_round'] = np.round(d['targets'] / 0.02) * 0.02
    
    max_mode_0_02_dic = df_pub.groupby('atoms_count_group')['targets_round'].agg(pd.Series.mode).to_dict()
    
    for key, val in max_mode_0_02_dic.items():
        max_mode_0_02_dic[key] = np.max(max_mode_0_02_dic[key])
    
    output = pr_test1['atoms_count_group'].map(max_mode_0_02_dic)
    
    
    output = output.rename('predictions')
    df_output = pd.DataFrame(output)
    df_output.update(df_priv_pred_gr10000)
    df_output.to_csv('./submission.csv', index_label='id')

    
    
    

if __name__ == '__main__':
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    main(config)