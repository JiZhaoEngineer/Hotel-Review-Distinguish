#!/usr/bin/env python
# coding: utf-8

##################
###Author: Ji Zhao
##################

import numpy as np
import csv
import math


pos_word_list = []
neg_word_list = []
with open('positive-words.txt') as pos_word:
    pos_word_reader = csv.reader(pos_word, delimiter = ' ')
    for row in pos_word_reader:
        # save word in a list
        pos_word_list.append(row[0])
with open('negative-words.txt') as neg_word:
    neg_word_reader = csv.reader(neg_word, delimiter = ' ')
    for row in neg_word_reader:
        # save word in a list
        neg_word_list.append(row[0])
pron_list = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]


def sdg(in_file):
    f_dict = {}
    f_key_weight = {}
    ind = 0
    learning_rate = 0.1
    with open(in_file) as f:
        f_reader = csv.reader(f, delimiter = '\t')
        for row in f_reader:
            feature1, feature2, feature3, feature4, feature5, feature6 = 0, 0, 0, 0, 0, 0
            # row[0] is identifier
            key = str(row[0])
            # row[1] is user's rating
            f_dict[ind] = row[1].lower().split()
            # extract feature of each review
            for i in range(len(f_dict[ind])):
                # feature 5: punctuation
                if '!' in f_dict[ind][i]:
                    feature5 += 1
                    f_dict[ind][i] = f_dict[ind][i].strip('!')
                if '.' in f_dict[ind][i]:
                    f_dict[ind][i] = f_dict[ind][i].strip('.')
                if ',' in f_dict[ind][i]:
                    f_dict[ind][i] = f_dict[ind][i].strip(',')
                if '\'' in f_dict[ind][i]:
                    f_dict[ind][i] = f_dict[ind][i].strip('\'')
                if '?' in f_dict[ind][i]:
                    f_dict[ind][i] = f_dict[ind][i].strip('?')
                if '\"' in f_dict[ind][i]:
                    f_dict[ind][i] = f_dict[ind][i].strip('\"')
                # feature 1: count positive words
                if f_dict[ind][i] in pos_word_list:
                    feature1 += 1
                # feature 2: count negative words
                if f_dict[ind][i] in neg_word_list:
                    feature2 += 1
                # feature 3: 'no'
                if 'no' in f_dict[ind]:
                    feature3 += 1
                # feature 4: count 1st and 2nd pronouns
                if f_dict[ind][i] in pron_list:
                    feature4 += 1
                # feature 6: count word of doc
                feature6 = math.log(len(f_dict[ind]))
            if feature3 > 0:
                feature3 = 1
            if feature5 > 0:
                feature5 = 1
            f_key_weight[ind] = [key, feature1, feature2, feature3, feature4, feature5, feature6, 1]
            ind += 1
    return f_key_weight


def write_to_file(output_file, y, feature):
    for ind in range(len(feature)):
        with open(output_file, 'a') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([feature[ind][0], feature[ind][1], feature[ind][2], feature[ind][3],                              feature[ind][4], feature[ind][5], round(feature[ind][6], 2), y])

def feature_true_y(y, feature):
    feature_y = {}
    for ind in range(len(feature)):
        feature_y[ind] = [feature[ind][1], feature[ind][2], feature[ind][3],                          feature[ind][4], feature[ind][5], feature[ind][6], 1, y]
    return feature_y


def random_train_data(pos_feature, neg_feature):
    feature = {}
    for i in range(len(pos_feature)):
        feature[i] = pos_feature[i]
    for i in range(len(pos_feature), len(pos_feature) + len(neg_feature)):
        feature[i] = neg_feature[i - len(pos_feature)]
    keys = list(feature.keys())
    np.random.shuffle(keys)
    feature_shuffle = [(key, feature[key]) for key in keys]
    train_number = (int)(0.8 * len(feature_shuffle))
    train_data =feature_shuffle[:train_number] 
    test_data = feature_shuffle[train_number:]
    return train_data, test_data


def update_weight_bias(w, x):
    w_update = [0, 0, 0, 0, 0, 0, 0]
    learning_rate = 0.1
    for i in range(len(x)):
        for j in range(len(w)):
            w_x_sum = np.dot(w, x[i][1][:-1])
            w_update[j] = (sigmoid(w_x_sum) - x[i][1][-1]) * x[i][1][j]
            w[j] = w[j] - learning_rate * w_update[j]
    return w


# calculate sigmoid
def sigmoid(x):
    try:
        ans = math.exp(-x)
    except OverflowError:
        ans = float('inf')
    return 1/(1 + ans)

# get the system output
# x: f1, f2, f3, f4, f5, f6, 1
def score(w, x):
    sum_w_x = np.dot(w, x)
    
    bundary = sigmoid(sum_w_x)
    return bundary


# test_data: index, f1, f2, f3, f4, f5, f6, 1, true_y
def accuracy(w, test_data):
    pos_true, pos_false, neg_true, neg_false = 0, 0, 0, 0
    for f in test_data:
        if score(w, f[1][:-1]) > 0.5:
            y_hat = 1
            if y_hat == f[1][-1]:
                pos_true += 1
            else:
                pos_false += 1
        else:
            y_hat = 0
            if y_hat == f[1][-1]:
                neg_true += 1
            else:
                neg_false += 1
    return (pos_true + neg_true)/len(test_data)


# In[22]:


# write to file
input_pos = 'hotelPosT-train.txt'
input_neg = 'hotelNegT-train.txt'
output_file = 'Zhao-Ji-assgn2-out.csv'
open(output_file,'w').close()
pos_key_feature = sdg(input_pos)
neg_key_feature = sdg(input_neg)
write_to_file(output_file, 1, pos_key_feature)
write_to_file(output_file, 0, neg_key_feature)

pos_feature_y = feature_true_y(1, pos_key_feature)
neg_feature_y = feature_true_y(0, neg_key_feature)
#f1, f2, f3, f4, f5, f6, 1, true_y

train_data, test_data = random_train_data(pos_feature_y, neg_feature_y)
len_train_data = len(train_data)
train_data1 = train_data[0:(int)(0.25*len_train_data)]
train_data2 = train_data[(int)(0.25*len_train_data):(int)(0.5*len_train_data)]
train_data3 = train_data[(int)(0.5*len_train_data):(int)(0.75*len_train_data)]
train_data4 = train_data[(int)(0.75*len_train_data):]
train_data5 = test_data
# print(len(train_data1),len(train_data2),len(train_data3),len(train_data4),len(train_data5))



#1
# train_data = train_data1 + train_data2 + train_data3 + train_data4
# del_data = test_data
train_data_v1 = train_data1 + train_data2 + train_data3 + train_data4
test_data_v1 = train_data5
w0, w1, w2, w3, w4, w5, bias = 0, 0, 0, 0, 0, 0, 0
w = [w0, w1, w2, w3, w4, w5, bias]
w_v1 = update_weight_bias(w, train_data_v1)
acc1 = accuracy(w_v1, test_data_v1)

# print(w_v1)
# print(acc1)


#2
train_data_v2 = train_data1 + train_data2 + train_data3 + train_data5
test_data_v2 = train_data4

w0, w1, w2, w3, w4, w5, bias = 0, 0, 0, 0, 0, 0, 0
w = [w0, w1, w2, w3, w4, w5, bias]
w_v2 = update_weight_bias(w, train_data_v2)
acc2 = accuracy(w_v2, test_data_v2)

#3
train_data_v3 = train_data1 + train_data2 + train_data4 + train_data5
test_data_v3 = train_data3

w0, w1, w2, w3, w4, w5, bias = 0, 0, 0, 0, 0, 0, 0
w = [w0, w1, w2, w3, w4, w5, bias]
w_v3 = update_weight_bias(w, train_data_v3)
acc3 = accuracy(w_v3, test_data_v3)

 w1, w2, w3, w4, w5, bias = 0, 0, 0, 0, 0, 0, 0
# w = [w0, w1, w2, w3, w4, w5, bias]
# w_v5 = update_weight_bias(w, train_data_v5)
# acc5 = accuracy(w_v5, test_data_v5)
#4
train_data_v4 = train_data1 + train_data3 + train_data4 + train_data5
test_data_v4 = train_data2

w0, w1, w2, w3, w4, w5, bias = 0, 0, 0, 0, 0, 0, 0
w = [w0, w1, w2, w3, w4, w5, bias]
w_v4 = update_weight_bias(w, train_data_v4)
acc4 = accuracy(w_v4, test_data_v4)

#5
train_data_v5 = train_data2 + train_data3 + train_data4 + train_data5
test_data_v5 = train_data1

w0,


# print(w,'\n', w_v2, '\n', w_v3,'\n', w_v4,'\n', w_v5)
# print(acc1, '\n', acc2, '\n',acc3, '\n',acc4, '\n',acc5)


test_input_file = 'HW2-testset.txt'
test_output_file = 'Zhao-Ji-assgn2-out.txt'
test_key_feature = sdg(test_input_file)

# train_data = train_data1 + train_data2 + train_data3 + train_data4 + test_data

w0, w1, w2, w3, w4, w5, bias = 0, 0, 0, 0, 0, 0, 0
w = [w0, w1, w2, w3, w4, w5, bias]
w = update_weight_bias(w, train_data)
open(test_output_file,'w').close()
  
def test_data_write_to_file(test_output_file, test_key_feature, w):  
    for ind in range(len(test_key_feature)):
        with open(test_output_file, 'a') as f:
            writer = csv.writer(f, delimiter='\t')
            if score(w, test_key_feature[ind][1:]) > 0.5:
                writer.writerow([test_key_feature[ind][0], 'POS'])
            else:
                writer.writerow([test_key_feature[ind][0], 'NEG'])
test_data_write_to_file(test_output_file, test_key_feature, w)





