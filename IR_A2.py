# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:21:05 2020

@author: Aditya Agarwal
"""

from pandas import DataFrame
import pandas as pd
import numpy as np
import time
import os
from random import randint
from PIL import Image, ImageTk
import tkinter as tk

print("---Welcome to LSH---\n ---Initializing---\n")
HEIGHT = 800
WIDTH = 800

### SHINGLING ###

def get_fileslist(destination_folder):
    """
    Returns list of files
    """
    listoffiles = []
    data = pd.read_csv(destination_folder,sep='\t')
    for row in range(len(data)):
        listoffiles.append(data.iat[row,0])
    global Total_Files
    Total_Files = len(listoffiles)
    return listoffiles

def matrix_create(lists,z=5):
    """
    Generates incidence-matrix for K-grams(Shingles)
    """
    df_temp = DataFrame(columns=[r for r in range(len(lists))])
    i=-1
    for h in lists:
            i+=1
            j=0
            while j<len(h)-z+1:
                rowshing = h[j:j+z]
                if (rowshing in df_temp.index) == False:
                    df_temp.loc[rowshing] = [0 for j in range(df_temp.shape[1])]
                df_temp.at[ rowshing, i ] = 1
                j+=1
    return df_temp
                 
def doShingling(direc,length_shingle=5):
    """
     Performs shingling and builds incidence matrix for shingles
    """
    lists = get_fileslist(direc)
    result_matrix = None
    result_matrix = matrix_create(lists,length_shingle)
    return result_matrix,lists

### Minhashing ###

def MinHashingNumpy(df_shingles,num_hashes=100):
    """
    This function generates random permutations of indices and then
    for each permutation it generates the signatures. Numpy is used
    to accelerate the process. The first occurence of a '1' for a
    shingle is used to generate the signature for a doc.
    """
    
    ListOL_NP = []
    rows = [i for i in range(len(df_shingles))]
    for _ in range(num_hashes):
        temp_permutation = np.random.permutation(rows)
        ListOL_NP.append(temp_permutation)
    list_signatures = []
    for j in range(num_hashes):
        cur_perm = df_shingles.iloc[ListOL_NP[j], :].values
        cur_sign = []
        for k in range(Total_Files):
            # returns the index of the first occurrence of the maximum value.
            cur_sign.append(cur_perm[:, k].argmax())
        list_signatures.append(cur_sign)
    return list_signatures


### Latent Semantic Hashing ###
    
def hashBand(cur_band,buckets_list):
    """
    Function to hash one band of the document given as input to one of the buckets
    in bucket list given as input.
    """
    for c in cur_band.columns:
        hs = hash(tuple(cur_band[c].values))
        if hs in buckets_list: 
            buckets_list[hs].append(c)
        else: 
            buckets_list[hs] = [c]
            
def doLSH(df_signature,rows_band=5):
    """
    Actual LSH starts here using the signature matrix. Documents in the same band that are similar 
    have higher probability of landing in the same bucket. Returns the buckets list which contains
    buckets for documents.
    """
    rows = df_signature.shape[0]
    bands = rows//rows_band
    buckets_dict= []
    for i in range(bands):
        buckets_dict.append(dict())
    for i in range(0, rows-rows_band+1, rows_band):
        band = df_signature.loc[i:i+rows_band-1,:]
        hashBand(band, buckets_dict[int(i/rows_band)])

    return buckets_dict

### QUERYING ###
    
def hashQueryBand(cur_band):
    """
    helper function to hash one band of the document given as input
    """
    hashes = []
    hs = hash(tuple(cur_band.values))
    hashes.append(hs)
    return hashes

def getSimDocs(cur_doc, buckets_dict, df_signature, rows_band):
    """
    Returns documents that are in the same bucket as the current document/query
    """
    rows = df_signature.shape[0]
    bands = rows//rows_band
    qbucketList = []
    for i in range(0, rows-rows_band+1, rows_band):
        band = df_signature.loc[i:i+rows_band-1, cur_doc]
        qbucketList.append(hashQueryBand(band))
        
    simDocs = set()
    for i in range(len(qbucketList)):
        for j in range(len(qbucketList[i])):
            simDocs.update(set(buckets_dict[i][qbucketList[i][j]]))

    return simDocs

def similarity_J(query, simDocs, df_signature):
    """
    This function finds jaccard similarity between two documents
    """
    # Jaq Sim formula : C1 AND C2 / C1 OR C2
    try:
        query = df_signature[query]
        simDocs = df_signature[simDocs]
        return sum(query & simDocs)/sum(query | simDocs)
    except:
        return 0
    
def similarity_Cosine(query, simDocs, df_signature):
    """
    This function finds cosine similarity between two documents
    """
    try:
        query = df_signature[query]
        simDocs = df_signature[simDocs]
        return np.dot(simDocs,query)/(np.sum(simDocs**2) * np.sum(query**2))**0.5
    except:
        return 0

def comp(item):
    """
    helper function to compare similarity score for sorting docs
    """
    return item[1]

def setShingles(ss):
    """
    helper function to set size of shingles
    """
    global ShingleSize
    ShingleSize = int(ss)
    
def setPerms(nps):
    """
    helper function to set number of permutations
    """
    global num_perms
    num_perms = int(nps)

def setBands(bns):
    """
    helper function to set number of bands
    """
    global num_bands
    num_bands = int(bns)

def getDir(direc):
    """
    Get directory input and start processing the documents.
    """
    ### Shingling ###
    global ShingleSize
    #print(ShingleSize)
    label['text'] = "---Shingling Started---"
    print("---Shingling Started---")
    crt = time.time()
    global df_shingles, all_files
    df_shingles, all_files = doShingling(direc,ShingleSize)
    print(df_shingles.head())
    label['text'] = "--- Shingling Completed. Time taken --- " + str(time.time()-crt)
    print("--- Shingling Completed. Time taken --- ",time.time()-crt)
    ### Minhashing ###
    global num_perms
    print("--- Numpy based Minhashing Started ---\n")
    ct = time.time()
    signs = MinHashingNumpy(df_shingles,num_perms)
    global df_signature
    df_signature=pd.DataFrame(signs)
    print(df_signature.head())
    print("--- Minhashing Numpy Completed. Time taken --- ",time.time()-ct)
    ### LSH ###
    global Total_Files
    global num_bands
    print("--- LSH Started ---")
    ct = time.time()
    global rows_band
    rows_band = df_signature.shape[0]//num_bands
    #print(rows_band)
    global bucks
    bucks = doLSH(df_signature,rows_band)
    print("--- LSH Completed. Time taken --- ",time.time()-ct)
    print(bucks)
    label['text'] = "--- Pre-processing, Shingling, Minhashing and LSH Completed. \n Total Time taken is : " + str(time.time()-crt)
    
def getQuery(query):
    """
    Get query sequence and process the query.
    """
    # now we wait for query
    try:
        global rows_band
        docid = all_files.index(query)
        print("doc id is : " + str(docid))
        ct = time.time()
        simDocs = getSimDocs(docid, bucks, df_signature, rows_band)
        # print(simDocs)
        ranked_list = []
        # get value of similarity using preferred metric
        for i in simDocs:
            if i == docid: 
                continue
            score = similarity_Cosine(docid, i, df_signature)
            ranked_list.append((i, score))
        print("--- Search Completed. Time taken --- ",time.time()-ct)
        ranked_list = sorted(ranked_list,reverse=True,key=comp)
        fin_str = f"Here are are the top similar documents for the query \n"
        cnt=0
        # top 10 documents based on similarity
        for ids, similarity in ranked_list:
            if cnt>9:
                break
            for idx in range(len(all_files)):
                if cnt>9:
                    break
                if ids == idx:
                    fin_str += (f"\n Document {idx} with score {similarity}")
                    cnt += 1
                    print(f"File {idx} with score {similarity}")
        fin_str += "\n\n Search Completed. Time taken is : " + str(time.time()-ct)
        label['text'] = fin_str
        ### We can set a threshold for max or min similarity and then retrieve docs 
        maxSim = 0.8
    except:
        # New document previously not in corpus. Need to process
        global Total_Files
        global ShingleSize
        global num_perms
        global num_bands
        print("Current document not in index\n")
        print("Therefore, processing Query\n")
        ct = time.time()
        global df_shingles_query
        df_shingles_query = matrix_create([query],ShingleSize)
        df_shingles_query.columns = [f'{Total_Files}']
        df_shingles_new = pd.concat([df_shingles,df_shingles_query], axis=1,sort=False)
        Total_Files+=1
        df_shingles_new = df_shingles_new.fillna(0)
        signs_new = MinHashingNumpy(df_shingles_new,num_perms)
        df_signature_new=pd.DataFrame(signs_new)
        bucks_new = doLSH(df_signature_new,df_signature_new.shape[0]//num_bands)
        simDocs_query = getSimDocs(Total_Files-1, bucks_new, df_signature_new, df_signature_new.shape[0]//num_bands)
        print(simDocs_query)
        ranked_list = []
        # get value of similarity using preferred metric
        for i in simDocs_query:
            if i == Total_Files-1: 
                continue
            score = similarity_Cosine(Total_Files-1, i, df_signature)
            ranked_list.append((i, score))
        # top 10 documents based on similarity
        ranked_list = sorted(ranked_list,reverse=True,key=comp)
        fin_str = f"Here are are the top similar documents for the query \n"
        cnt=0
        for ids, similarity in ranked_list:
            if cnt>9:
                break
            for idx in range(Total_Files):
                if cnt>9:
                    break
                if ids == idx and similarity>0:
                    cnt += 1
                    fin_str += (f"\n Document {idx} with score {similarity}")
                    print(f"File {idx} with score {similarity}")
        fin_str += "\n\n Search Completed. Time taken is : " + str(time.time()-ct)
        if cnt==0:
            fin_str += "\n\n Sorry. No similar documents found by LSH."
        label['text'] = fin_str
        Total_Files-=1
    print("--- THE END ---\n")
    
#direc = input("Enter directory of txt file\n")
#getDir(direc)

### Begin GUI Tkinter ###

root = tk.Tk()

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

imagex = Image.open('apples.jpg')
photo = ImageTk.PhotoImage(imagex,master=root)
background_label = tk.Label(root, image=photo)
background_label.image = photo
background_label.place(relwidth=1, relheight=1)
frame = tk.Frame(root, bg='#C0C0C0', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.90, relheight=0.065, anchor='n')

entry = tk.Entry(frame, font=40)
entry.place(relwidth=0.65, relheight=0.70)

button = tk.Button(frame, text="Enter corpus path", font=30, command=lambda: getDir(entry.get()))
button.place(relx=0.7, relheight=0.80, relwidth=0.3)

frame2 = tk.Frame(root, bg='#C0C0C0', bd=5)
frame2.place(relx=0.5, rely=0.22, relwidth=0.90, relheight=0.065, anchor='n') 

entry2 = tk.Entry(frame2, font=40)
entry2.place(relwidth=0.65, relheight=0.70)

button2 = tk.Button(frame2, text="Enter Query", font=30, command=lambda: getQuery(entry2.get()))
button2.place(relx=0.7, relheight=0.80, relwidth=0.3)

frame3 = tk.Frame(root, bg='#C0C0C0', bd=5)
frame3.place(relx=0.2, rely=0.34, relwidth=0.30, relheight=0.065, anchor='n') 

entry3 = tk.Entry(frame3, font=40)
entry3.place(relwidth=0.4, relheight=0.80)

button3 = tk.Button(frame3, text="Shingle Size", font=25, command=lambda: setShingles(entry3.get()))
button3.place(relx=0.42, relheight=0.90, relwidth=0.6)

frame4 = tk.Frame(root, bg='#C0C0C0', bd=5)
frame4.place(relx=0.5, rely=0.34, relwidth=0.30, relheight=0.065, anchor='n') 

entry4 = tk.Entry(frame4, font=40)
entry4.place(relwidth=0.4, relheight=0.80)

button4 = tk.Button(frame4, text="Num Perms", font=25, command=lambda: setPerms(entry4.get()))
button4.place(relx=0.42, relheight=0.90, relwidth=0.6)

frame5 = tk.Frame(root, bg='#C0C0C0', bd=5)
frame5.place(relx=0.81, rely=0.34, relwidth=0.30, relheight=0.065, anchor='n') 

entry5 = tk.Entry(frame5, font=40)
entry5.place(relwidth=0.4, relheight=0.80)

button5 = tk.Button(frame5, text="Num Bands", font=25, command=lambda: setBands(entry5.get()))
button5.place(relx=0.42, relheight=0.90, relwidth=0.6)

lower_frame = tk.Frame(root, bg='#C0C0C0', bd=10)
lower_frame.place(relx=0.5, rely=0.5, relwidth=0.9, relheight=0.4, anchor='n')

label = tk.Label(lower_frame)
label.place(relwidth=1, relheight=1)

root.mainloop()