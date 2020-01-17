from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | )
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        
        O = []
        Osequence = Osequence.astype(str)
        for i in Osequence:
            O.append(self.obs_dict[i])
        ###################################################
        # Edit here
        ###################################################
        
        for j in range(S):
            alpha[j, 0] = self.pi[j] * self.B[j, O[0]]
        
        t = 1
        
        while t < L:
            j = 0
            while j < S:                                
                temp = []                
                k = 0
                while k < S:
                    temp.append(self.A[k, j] * alpha[k, t - 1])
                    var = np.sum(temp)
                    k += 1                
                alpha[j, t] = self.B[j, O[t]] * var
                j += 1
            t += 1
       
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, )
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        O = []
        Osequence = Osequence.astype(str)
        for i in Osequence:
            O.append(self.obs_dict[i])
        ###################################################
        # Edit here
        ###################################################
        j=0
        while j < S:
        #for j in range(S):
            beta[j, L - 1] = 1
            j +=1
        t = L-2
        while t > -1:
            i = 0
            while i < S:
                temp=[]
                j = 0
                while j < S:
                    temp.append([beta[j, t + 1] * self.A[i, j] * self.B[j, O[t + 1]]])                    
                    j += 1
                beta[i,t] = np.sum(temp)
                i += 1
            t -= 1
        
        
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | )
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:, -1])
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, )
        """
        prob = np.zeros((len(self.pi),len(Osequence)))
        '''
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        O = []
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        Osequence = Osequence.astype(str)
        for i in Osequence:
            O.append(self.obs_dict[i])
        ###################################################
        # Edit here
        ###################################################
        N = self.A.shape[0]
        denominator = sum(alpha[:, -1])
        
        for t in reversed(range(L - 1)):
            for i in range(S):
                prob[i, t] = (np.multiply(alpha[i,t],beta[i,t])/denominator)
        '''
        fwd_array,bwd_array,seq_prob,i=self.forward(Osequence),self.backward(Osequence),self.sequence_prob(Osequence),0
       
       
        
        counter1 = len(self.pi)
        counter2 = len(Osequence)
        while i < counter1:
            j = 0
            while j < counter2:
                var1 = fwd_array[i][j]*bwd_array[i][j]
                var2 = var1/seq_prob
                prob[i][j] = var2
                j += 1
            i += 1
        return prob
    
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, )
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
       
        ###################################################
        # Edit here
        ###################################################
        O = []
        Osequence = Osequence.astype(str)
        for i in Osequence:
            O.append(self.obs_dict[i])
        
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        f_array=self.forward(Osequence)
        b_array=self.backward(Osequence)
        seq_prob=self.sequence_prob(Osequence)
        for i in range(len(self.pi)):
            
           
            for j in range(len(self.pi)):
                for k in range(len(Osequence)-1):
                    #prob[i][j][k] = (self.A[i][j]*self.B[i][O[k+1]]*f_array[i][k]*b_array[j][self.obs_dict[Osequence[k+1]]])/seq_prob
                    prob[i][j][k] = (self.A[i][j]*self.B[j][[self.obs_dict[Osequence[k+1]]]]*f_array[i][k]*b_array[j][k+1])/seq_prob
                j+=1    
        
          
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        ###################################################
        
        
        
        state_dict_inverse = {}
        
        for k,v in self.state_dict.items():
            state_dict_inverse[v] = k
         
        
        ###################################################
        # Edit here
        S,L = len(self.pi),len(Osequence)
        
        interm_path = []
        for i in range(L):
            interm_path.append(0)
        #print('interm_path',interm_path)
        delta = []
        for i in range(S):
            temp = []
            for j in range(L):
                temp.append(float(0))
            delta.append(temp)
        
        delta = np.asarray(delta)
        
        backtrack_matrix = []
        for i in range(S):
            temp = []
            for j in range(L):
                temp.append(float(0))
            backtrack_matrix.append(temp)
        
        backtrack_matrix = np.asarray(backtrack_matrix)
        
        i=0
        while i < S:
            index_var = self.obs_dict[Osequence[0]]
            delta[i][0]=self.pi[i]*self.B[i][index_var]
            i += 1
            
        i = 1
        while i < L:
            j = 0
            while j < S:
                max_val,max_index,k=float("-inf"),-1,0
                while k < S:
                    cmp_var = self.obs_dict[Osequence[i]]
                    max_var = self.B[j][cmp_var]*self.A[k][j]*delta[k][i-1]
                    if max_val < max_var:
                        max_val,max_index = max_var,k
                        
                    k += 1
                delta[j][i], backtrack_matrix[j][i]=max_val,max_index
                
                j += 1
            i += 1
        
                    
        max_val,i=float("-inf"),0
        i=0
         
        while i < S:
            
        #for i in range(S):
            cpm_var = delta[i][L-1]
            if max_val < cpm_var:
                max_val,max_index = cpm_var,i
                
            i += 1    
        interm_path[L-1]=max_index
        
        i = L-2
        while (i > -1):
            index_v = int(interm_path[i+1])
            interm_path[i]=backtrack_matrix[index_v][i+1]
            i -= 1
            
        
            
            
        counter1 = len(interm_path)
        i = 0
        while i <counter1:
            inter_var = interm_path[i]
            inter_var2 = state_dict_inverse[inter_var]
            path.append(inter_var2)
            i +=1
        #for i in range(len(temp_path)):
            #path.append(state_dict_inverse[temp_path[i]])
        ###################################################
        return path

