# Information Retrieval Assignment-2
# Topic: Locality Sensitive Hashing
# No. of Group Members = 1
# Name: Navdeep Singh Narsinghia
# ID: 2017B5A71675H

import os
import nltk
import numpy as np
import math
import time
import random as rand
from operator import itemgetter
from nltk.corpus import stopwords

start_time = time.time()    # start timer

#scanning the dataset and extracting content
def getText(filename):
    names.append(filename)
    doc = open(filename, encoding="utf8", errors='ignore')
    fullText = []
    for line in doc:
        fullText.append(line)
    return '\n'.join(fullText)

#building set of k-shingles from each file
def building_shingles_set(docs, shingles_length):
    shingles = []

    for i in range(len(docs)):
        doc = docs[i]
        doc, sh = build_shingles(doc, shingles_length)
        docs[i] = doc
        shingles.append(sh)
    return docs, shingles

#helper function to build set of k_shingles
def build_shingles(doc, shingles_length):
    doc = doc.lower()
    doc = ''.join(doc.split(' '))
    shingles = {}
    for i in range(len(doc)):
        sub_string = ''.join(doc[i:i + shingles_length])
        if len(sub_string) == shingles_length and sub_string not in shingles:
            shingles[sub_string] = 1
    return doc, shingles.keys()

#building the document shingle matrix of 0s and 1s by checking if shingle is present in the document
def build_shingle_doc_matrix(whole_document, shingles):
    i = 0
    rows = {}
    for j in shingles:
        for k in j:
            if k not in rows:
                rows[k] = i
                i += 1

    shingle_doc_matrix =  np.zeros((len(rows), len(whole_document)))

    for j in range(len(whole_document)):
        for i in rows:
            if i in whole_document[j]:
                shingle_doc_matrix[rows[i], j] = 1
    return shingle_doc_matrix

#building signature matrix using minhashing technique w
def minhashing(shingle_doc_matrix, n):
    hash_functions = get_hashfunction(n)
    hash_values = []
    for f in hash_functions:
        value = [(i*f[0] + f[1])% shingle_doc_matrix.shape[0] for i in range(shingle_doc_matrix.shape[0])]
        hash_values.append(value)

    signature_matrix = np.zeros((n, shingle_doc_matrix.shape[1])) + float('inf')

    for i in range(shingle_doc_matrix.shape[1]):
        for j in range(shingle_doc_matrix.shape[0]):
            if shingle_doc_matrix[j, i] != 0:
                for k in range(n):
                    h = hash_values[k]
                    signature_matrix[k, i] = min(signature_matrix[k, i], h[j])
    return signature_matrix


def get_hashfunction(n):
    hashfunction = []
    for i in range(n):
        x = rand.randint(0, 10000)
        c = rand.randint(0, 10000)
        hashfunction.append([x, c])
    return hashfunction

#Applying LSH technique by dividing into bands
def locality_hashing(signature_matrix ,  bands, rows):
    global len_buckets
    array_buckets = []
    for band in range(bands):
        array_buckets.append([[] for i in range(len_buckets)])
    similarity_values = {}
    i = 0
    for b in range(bands):
        buckets = array_buckets[b]
        band = signature_matrix[i:i + rows, :]
        for col in range(band.shape[1]):
            k = int(sum(band[:, col]) % len(buckets))
            buckets[k].append(col)
        i = i + rows

        for f in buckets:
            if len(f) > 0:
                for f1 in f:
                    for g in buckets:
                        if len(g) > 0 :
                            for g1 in g:
                                if f1 != g1:
                                    pair = (min(f1, g1), max(f1, g1))
                                    if pair not in similarity_values:
                                        x = signature_matrix[:, f1]
                                        y = signature_matrix[:, g1]
                                        similarity = 1 - nltk.cluster.util.cosine_distance(x, y)  # cosine distance calculation
                                        similarity_values[pair] = similarity

    similarity_result = sorted(similarity_values.items(), key=itemgetter(1), reverse=True)
    return similarity_result

print("Welcome to the Locality Sensitive Hashing Program")
print("Please wait.....Processing\n")
shingles_length = 6                 # length of shingles
names = []                          #stores names of all files
docs = []                           #stores each file's contents

# scanning through training files
os.chdir('D:\\corpus\\')
for filename in os.listdir():
    if filename.endswith('.txt'):
        filename = getText(filename)
        docs.append(filename)

corpuslen = len(names)              # no. of training files training

#scanning through query
os.chdir('D:\\query\\')
for filename in os.listdir():
    if filename.endswith('.txt'):
        filename = getText(filename)
        docs.append(filename)


querylen = len(names)-corpuslen      #no. of queries

whole_document, shingles = building_shingles_set(docs[:], shingles_length)    #making shingles from document

len_buckets = 100
hash_table = [[] for i in range(len_buckets)]

shingle_doc_matrix = build_shingle_doc_matrix(whole_document, shingles)     #checking shingle present in doc or not

signature_matrix = minhashing(shingle_doc_matrix, 50)   #building signature matrix , n = no. of hash functions

similarity_result = locality_hashing(signature_matrix,  5, 10)       #(no. of bands, no. of rows)

x=0
for j in range(corpuslen, len(names), 1):
    print("Similarity report for " , names[j])
    print("\nDOCUMENT NAME                SIMILARITY")
    for i in range(len(similarity_result)):
        pair = similarity_result[i][0]
        if names[j] == names[pair[0]]:
            x = x+1
            print(names[pair[1]],"           ", similarity_result[i][1] * 100, '%\n' )
        elif names[j] == names[pair[1]]:
            print(names[pair[0]] ,"           ", similarity_result[i][1] * 100, '%\n' )
            x=x+1
        if x == 9:
            break

total_time = time.time()-start_time;

print("Time = " , total_time, "s")

