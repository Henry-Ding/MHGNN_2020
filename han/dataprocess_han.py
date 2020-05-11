# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from scipy import sparse

import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
from tools import *

class input_data_han(object):
    def __init__(self, args):
        self.args = args

        a_p_list_train = [[] for k in range(self.args.A_n)]
        p_a_list_train = [[] for k in range(self.args.P_n)]
        p_p_cite_list_train = [[] for k in range(self.args.P_n)]
        v_p_list_train = [[] for k in range(self.args.V_n)]
        p_v_list_train = [[] for k in range(self.args.P_n)]

        relation_f = ["a_p_list_train.txt", "p_a_list_train.txt",\
                              "p_p_citation_list.txt", "v_p_list_train.txt"]

        #store academic relational data 
        for i in range(len(relation_f)):
            f_name = relation_f[i]
            neigh_f = open(self.args.data_path + f_name, "r")
            for line in neigh_f:
                line = line.strip()
                node_id = int(re.split(':', line)[0])
                neigh_list = re.split(':', line)[1]
                neigh_list_id = re.split(',', neigh_list)
                if f_name == 'a_p_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
                elif f_name == 'p_a_list_train.txt':
                    for j in range(len(neigh_list_id)):
                        p_a_list_train[node_id].append('a'+str(neigh_list_id[j]))
                elif f_name == 'p_p_citation_list.txt':
                    for j in range(len(neigh_list_id)):
                        p_p_cite_list_train[node_id].append('p'+str(neigh_list_id[j]))
                else:
                    for j in range(len(neigh_list_id)):
                        v_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
            neigh_f.close()

        #store paper venue 
        p_v = [0] * self.args.P_n
        p_v_f = open(self.args.data_path + 'p_v.txt', "r")
        for line in p_v_f:
            line = line.strip()
            p_id = int(re.split(',',line)[0])
            v_id = int(re.split(',',line)[1])
            p_v[p_id] = v_id
        p_v_f.close()
        
        a_class = [0] * self.args.A_n
        a_class_f = open(self.args.data_path + 'a_class.txt', "r")
        for line in a_class_f:
            line = line.strip()
            a_id = int(re.split(',',line)[0])
            class_id = int(re.split(',',line)[1])
            a_class[a_id] = class_id
        a_class_f.close()
        
        train_mask = [0] * self.args.A_n
        train_mask_f = open(self.args.data_path + 'a_class_train.txt', "r")
        for line in train_mask_f:
            line = line.strip()
            a_id = int(re.split(',',line)[0])
            train_mask[a_id] = 1
        train_mask_f.close()
                
        test_mask = [0] * self.args.A_n
        test_mask_f = open(self.args.data_path + 'a_class_test.txt', "r")
        for line in test_mask_f:
            line = line.strip()
            a_id = int(re.split(',',line)[0])
            test_mask[a_id] = 1
        test_mask_f.close()        
        
        #paper neighbor: author + citation + venue
        p_list_train = [[] for k in range(self.args.P_n)]
        for i in range(self.args.P_n):
            p_list_train[i] += p_a_list_train[i]
            p_list_train[i] += p_p_cite_list_train[i] 
            p_list_train[i].append('v' + str(p_v[i]))

        self.a_p_list_train =  a_p_list_train
        self.p_a_list_train = p_a_list_train
        self.p_p_cite_list_train = p_p_cite_list_train
        self.p_list_train = p_list_train
        self.v_p_list_train = v_p_list_train


        #store paper content pre-trained embedding
        p_abstract_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        p_a_e_f = open(self.args.data_path + "p_abstract_embed.txt", "r")
        for line in islice(p_a_e_f, 1, None):
            values = line.split()
            index = int(values[0])
            embeds = np.asarray(values[1:], dtype='float32')
            p_abstract_embed[index] = embeds
        p_a_e_f.close()

        p_title_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        p_t_e_f = open(self.args.data_path + "p_title_embed.txt", "r")
        for line in islice(p_t_e_f, 1, None):
            values = line.split()
            index = int(values[0])
            embeds = np.asarray(values[1:], dtype='float32')
            p_title_embed[index] = embeds
        p_t_e_f.close()

        self.p_abstract_embed = p_abstract_embed
        self.p_title_embed = p_title_embed

        #store pre-trained network/content embedding
        a_net_embed = np.zeros((self.args.A_n, self.args.in_f_d))
        p_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        v_net_embed = np.zeros((self.args.V_n, self.args.in_f_d)) 
        net_e_f = open(self.args.data_path + "node_net_embedding.txt", "r")
        for line in islice(net_e_f, 1, None):
            line = line.strip()
            index = re.split(' ', line)[0]
            if len(index) and (index[0] == 'a' or index[0] == 'v' or index[0] == 'p'):
                embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
                if index[0] == 'a':
                    a_net_embed[int(index[1:])] = embeds
                elif index[0] == 'v':
                    v_net_embed[int(index[1:])] = embeds
                else:
                    p_net_embed[int(index[1:])] = embeds
        net_e_f.close()
        #for i in range(a_net_embed.shape[0]):
            #if (a_net_embed[i]==[0]*a_net_embed.shape[1]).all():
                #print(i)
        #v的embedding作为p的embedding
        p_v_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        p_v = [0] * self.args.P_n
        p_v_f = open(self.args.data_path + "p_v.txt", "r")
        for line in p_v_f:
            line = line.strip()
            p_id = int(re.split(',', line)[0])
            v_id = int(re.split(',', line)[1])
            p_v[p_id] = v_id
            p_v_net_embed[p_id] = v_net_embed[v_id]
            p_v_list_train[p_id].append('v'+str(v_id))

        p_v_f.close()
        #a的embedding平均作为p的embedding

        p_a_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        for i in range(self.args.P_n):
            if len(p_a_list_train[i]):
                for j in range(len(p_a_list_train[i])):
                    a_id = int(p_a_list_train[i][j][1:])
                    p_a_net_embed[i] = np.add(p_a_net_embed[i], a_net_embed[a_id])
                p_a_net_embed[i] = p_a_net_embed[i] / len(p_a_list_train[i])

        p_ref_net_embed = np.zeros((self.args.P_n, self.args.in_f_d))
        for i in range(self.args.P_n):
            if len(p_p_cite_list_train[i]):
                for j in range(len(p_p_cite_list_train[i])):
                    p_id = int(p_p_cite_list_train[i][j][1:])
                    p_ref_net_embed[i] = np.add(p_ref_net_embed[i], p_net_embed[p_id])
                p_ref_net_embed[i] = p_ref_net_embed[i] / len(p_p_cite_list_train[i])
                ##尝试增加自己本身
                #p_ref_net_embed[i] = (p_net_embed[i]+p_ref_net_embed[i]) / (len(p_p_cite_list_train[i])+1)

            else:
                p_ref_net_embed[i] = p_net_embed[i]

        #用平均embedding代替			
        a_text_embed = np.zeros((self.args.A_n, self.args.in_f_d))
        for i in range(self.args.A_n):
            if len(a_p_list_train[i]):
                for j in range(len(a_p_list_train[i])):       
                    p_id = int(a_p_list_train[i][j][1:])
                    a_text_embed[i] = np.add(a_text_embed[i], p_abstract_embed[p_id])
                a_text_embed[i] = a_text_embed[i] / len(a_p_list_train[i])        

                

        #用平均embedding代替
        v_text_embed = np.zeros((self.args.V_n, self.args.in_f_d))
        for i in range(self.args.V_n):
            if len(v_p_list_train[i]):
                for j in range(len(v_p_list_train[i])):       
                    p_id = int(v_p_list_train[i][j][1:])
                    v_text_embed[i] = np.add(v_text_embed[i], p_abstract_embed[p_id])
                v_text_embed[i] = v_text_embed[i] / len(v_p_list_train[i])              
               
        #APA_data=[]
        #APA_row=[]
        #APA_col=[]
        #APA_f = open(self.args.data_path + 'APA.txt', "r")
        #for line in APA_f:
            #line = line.strip()
            #row_id = int(re.split(',',line)[0])
            #col_id = int(re.split(',',line)[1])
            #APA_row.append(row_id)
            #APA_col.append(col_id)
            #APA_data.append(int(1))            
        #APA_f.close()
        #APA_matrix=sparse.csr_matrix((APA_data, (APA_row, APA_col)), shape=(self.args.A_n, self.args.A_n))
        
        #APVPA_data=[]
        #APVPA_row=[]
        #APVPA_col=[]
        #APVPA_f = open(self.args.data_path + 'APVPA.txt', "r")
        #for line in APVPA_f:
            #line = line.strip()
            #row_id = int(re.split(',',line)[0])
            #col_id = int(re.split(',',line)[1])
            #APVPA_row.append(row_id)
            #APVPA_col.append(col_id)
            #APVPA_data.append(int(1))            
        #APVPA_f.close()
        #APVPA_matrix=sparse.csr_matrix((APVPA_data, (APVPA_row, APVPA_col)), shape=(self.args.A_n, self.args.A_n))        
        
        #APPA_data=[]
        #APPA_row=[]
        #APPA_col=[]
        #APPA_f = open(self.args.data_path + 'APPA.txt', "r")
        #for line in APPA_f:
            #line = line.strip()
            #row_id = int(re.split(',',line)[0])
            #col_id = int(re.split(',',line)[1])
            #APPA_row.append(row_id)
            #APPA_col.append(col_id)
            #APPA_data.append(int(1))            
        #APPA_f.close()
        #APPA_matrix=sparse.csr_matrix((APPA_data, (APPA_row, APPA_col)), shape=(self.args.A_n, self.args.A_n))        
                
        #sparse.save_npz(self.args.data_path +'APA_matrix.npz', APA_matrix) 
        #sparse.save_npz(self.args.data_path +'APVPA_matrix.npz', APVPA_matrix) 
        #sparse.save_npz(self.args.data_path +'APPA_matrix.npz', APPA_matrix) 
        
        APA_matrix = sparse.load_npz(self.args.data_path +'APA_matrix.npz')        
        APVPA_matrix = sparse.load_npz(self.args.data_path +'APVPA_matrix.npz')        
        APPA_matrix = sparse.load_npz(self.args.data_path +'APPA_matrix.npz')        
        
        self.p_v = p_v
        self.p_v_list_train=p_v_list_train
        self.p_v_net_embed = p_v_net_embed
        self.p_a_net_embed = p_a_net_embed
        self.p_ref_net_embed = p_ref_net_embed
        self.p_net_embed = p_net_embed
        self.a_net_embed = a_net_embed
        self.a_text_embed = a_text_embed
        self.v_net_embed = v_net_embed
        self.v_text_embed = v_text_embed
        self.a_class = a_class
        self.train_mask = train_mask
        self.test_mask = test_mask
        
        #self.meta_path()
        
        self.APA_matrix = APA_matrix
        self.APVPA_matrix = APVPA_matrix
        self.APPA_matrix = APPA_matrix
        
        
        #self.val_mask = val_mask
        
    def meta_path(self):
        #定义基于元路径的邻居
        #A:
        #APA
        #AP APP 
        #APV APPV
        #self.APA = meta_path_count('APA', [self.args.A_n, self.args.P_n], self.a_p_list_train, self.p_a_list_train)

        #self.APVPA = meta_path_count('APVPA', [self.args.A_n, self.args.P_n, self.args.V_n, self.args.P_n], self.a_p_list_train, self.p_v_list_train,\
                        #self.v_p_list_train, self.p_a_list_train)
        self.APPA = meta_path_count('APPA', [self.args.A_n, self.args.P_n, self.args.P_n], self.a_p_list_train, \
                        self.p_p_cite_list_train, self.p_a_list_train)
        #self.AP = self.a_p_list_train
        #self.APP = meta_path_count('APP', [self.args.A_n, self.args.P_n], self.a_p_list_train, self.p_p_cite_list_train)
        #self.APV = meta_path_count('APV', [self.args.A_n, self.args.P_n], self.a_p_list_train, self.p_v_list_train)
        #self.APPV = meta_path_count('APPV', [self.args.A_n, self.args.P_n, self.args.P_n], self.a_p_list_train, self.p_p_cite_list_train, self.p_v_list_train)


        ##P:
        ##PA			
        ##PP PAP PVP
        ##PV PPV
        #self.PA = self.p_a_list_train
        #self.PP = self.p_p_cite_list_train
        #self.PAP = meta_path_count('PAP', [self.args.P_n, self.args.A_n], self.p_a_list_train, self.a_p_list_train)
        #self.PVP = meta_path_count('PVP', [self.args.P_n, self.args.V_n], self.p_v_list_train, self.v_p_list_train)			
        #self.PV = self.p_v_list_train			
        #self.PPV = meta_path_count('PPV', [self.args.P_n, self.args.P_n], self.p_p_cite_list_train, self.p_v_list_train)			

        ##V:
        ##VPA			
        ##VP VPP
        ##VPPV
        #self.VPA = meta_path_count('VPA', [self.args.V_n, self.args.P_n], self.v_p_list_train, self.p_a_list_train)
        #self.VP = self.v_p_list_train			
        #self.VPP = meta_path_count('VPP', [self.args.V_n, self.args.P_n], self.v_p_list_train, self.p_p_cite_list_train)			
        #self.VPPV = meta_path_count('VPPV', [self.args.V_n, self.args.P_n, self.args.P_n], self.v_p_list_train, self.p_p_cite_list_train, self.p_v_list_train)				
        #APA_f = open(self.args.data_path + "APA.txt", "w")
        #for i in range(len(self.APA)): 
            #if len(self.APA[i]):
                #for j in range(len(self.APA[i])):
                    #APA_f.write(str(i) + ","+self.APA[i][j][1:]+"\n")
        #APA_f.close()
        
        #APVPA_f = open(self.args.data_path + "APVPA.txt", "w")
        #for i in range(len(self.APVPA)): 
            #if len(self.APVPA[i]):
                #for j in range(len(self.APVPA[i])):
                    #APVPA_f.write(str(i) + ","+self.APVPA[i][j][1:]+"\n")
        #APVPA_f.close()
        
        APPA_f = open(self.args.data_path + "APPA.txt", "w")
        for i in range(len(self.APPA)): 
            if len(self.APPA[i]):
                for j in range(len(self.APPA[i])):
                    APPA_f.write(str(i) + ","+self.APPA[i][j][1:]+"\n")
        APPA_f.close()

if __name__ == '__main__':
    args = read_args()
    data = input_data_han(args)
    data.a_text_embed
    
    
    
