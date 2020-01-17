import numpy as np

from util import accuracy
from hmm import HMM

def get_ksi(tags_list,ksi):
    
    i = 0
    counter_var = len(tags_list)-1
    while i < (counter_var):
        counter1 = tags_list[i]
        counter2 = tags_list[i+1]
        if (counter1 in ksi) and (counter2 in ksi[tags_list[i]]):
           
            ksi[counter1][counter2]+=1
        elif (counter1 in ksi) and (counter2 not in ksi[counter1]):
            ksi[counter1][counter2]=1
        else:
           
            temp = {}
            temp[counter2]=1
            ksi[counter1]=temp
            
        i += 1
    return ksi
        

def get_delta(viterbi_param, gamma_param, ksi, training_data):
    i = 0
    data_len = len(training_data)
    while i < data_len:
        for w,t in zip(training_data[i].words,training_data[i].tags):
            if (t in gamma_param) and w in viterbi_param[t]:
                gamma_param[t] = gamma_param[t] + 1
                viterbi_param[t][w]= viterbi_param[t][w] + 1
            elif (t in gamma_param) and w not in viterbi_param[t]:
                gamma_param[t]= gamma_param[t] + 1
                viterbi_param[t][w]=1
            else:
                gamma_param[t]=1
                temp = {}
                temp[w]=1
                
                viterbi_param[t]=temp
        ksi_val = get_ksi(training_data[i].tags,ksi)
        i += 1
    return viterbi_param, gamma_param, ksi_val


def get_A(ksi,tags_list,tag_counter,A):
    
    i = 0
    while i < len(tags_list):
        counter1 = tags_list[i]
        if counter1 in ksi:
            j = 0
            while j < len(tags_list):
                counter2 = tags_list[j]
                if counter2 not in ksi[counter1]:
                    A[tag_counter[counter1]][tag_counter[counter2]]=0.0                    
                else:
                    den = sum(ksi[counter1].values())
                    A[tag_counter[counter1]][tag_counter[counter2]]=(ksi[counter1][counter2])/den
                j += 1
        else:
            A[tag_counter[counter1]][tag_counter[counter2]]=0.0                
        i += 1
    return A


def get_B(delta,tags_list,tag_counter,word_index,B):
    i = 0
    while i < len(tags_list):
        counter1 = tags_list[i]
        if counter1 in delta:
            for idx in word_index:
                counter2 = word_index[idx]
                if idx not in delta[counter1]:
                    B[tag_counter[counter1]][counter2]=0.0           
                else:
                    den = sum(delta[counter1].values())
                    B[tag_counter[counter1]][counter2]=(delta[counter1][idx])/den
        else:
            B[tag_counter[counter1]][counter2]=0.0
        i += 1
    
    return B



def get_pi(gamma_param,tags_list,tag_counter,pi):
    i = 0
    while i < len(tags_list):
        counter1 = tags_list[i]
        if counter1 in gamma_param:
            len_var = len(tags_list)
            pi[tag_counter[counter1]] = (gamma_param[counter1])/len_var
        i += 1        
    return pi
    
# TODO:
def model_training(training_data, tags_list):
    model, word_counter, gamma,delta, ksi,index,j = None,{},{},{},{},0,0
    
    
    while j < len(training_data):
        for w in training_data[j].words:
            if w not in word_counter:
                word_counter[w] = index
                index = index + 1
        j += 1
    

    delta, gamma, ksi = get_delta(delta, gamma, ksi, training_data)
    
    A,B,pi = [],[],[]
    
    for i in range(len(tags_list)):
        temp = []
        for j in range(len(tags_list)):
            temp.append(float(0))
        A.append(temp)
    
    for i in range(len(tags_list)):
        temp = []
        for j in range(len(word_counter)):
            temp.append(float(0))
        B.append(temp)
    
    for i in range(len(tags_list)):
        pi.append(float(0))
    
    A = np.asarray(A)
    B = np.asarray(B)
    pi = np.asarray(pi)
    
    tag_dict={}
    index=0
        
    for tag in tags_list:
        if tag not in tag_dict:
            tag_dict[tag]=index
        index= index + 1
        
    pi = get_pi(gamma,tags_list,tag_dict,pi)
    A = get_A(ksi,tags_list,tag_dict,A)
    B = get_B(delta,tags_list,tag_dict,word_counter,B)

    
    model=HMM(pi,A,B,word_counter,tag_dict)
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    tagging = []
    j = 0
    while j < len(test_data):
        k = 0
        counter2 = len(test_data[j].words)
        while k < counter2:
            
            if test_data[j].words[k] not in model.obs_dict:
                inter_var1 = list(model.obs_dict.values())
                val = max(inter_var1)+1
                index_counter = test_data[j].words[k]
                model.obs_dict[index_counter] = val
                len_obsdict = len(model.obs_dict)-1
                exp_var = 10**(-6)
                model.B=np.insert(model.B, len_obsdict, exp_var, axis=1)
            k += 1
        j += 1
                
   
    i = 0
    while i < len(test_data):
        counter1 = test_data[i].words
        tagging.append(model.viterbi(counter1))
        i += 1
    
    return tagging
