"""
@Author Syed Hasan Raza
CS461 Machine Learning Course
"""

import pandas as pd
import math
import numpy as np

#Loading Necessary files as dataframes using pandas
train_features = pd.read_csv("C:/Users/Hassan/Desktop/Intro to ML/HW_1/GitHub_Project/Twitter-Sentiment-Analysis/Dataset/question-4-train-features.csv", header = None)
train_labels = pd.read_csv("C:/Users/Hassan/Desktop/Intro to ML/HW_1/GitHub_Project/Twitter-Sentiment-Analysis/Dataset/question-4-train-labels.csv", header = None)
test_labels = pd.read_csv("C:/Users/Hassan/Desktop/Intro to ML/HW_1/GitHub_Project/Twitter-Sentiment-Analysis/Dataset/question-4-test-features.csv", header = None)
test_features = pd.read_csv("C:/Users/Hassan/Desktop/Intro to ML/HW_1/GitHub_Project/Twitter-Sentiment-Analysis/Dataset/question-4-test-labels.csv", header = None)
vocab = pd.read_csv('C:/Users/Hassan/Desktop/Intro to ML/HW_1/GitHub_Project/Twitter-Sentiment-Analysis/Dataset/question-4-vocab.txt', delimiter = '\t', header = None)

pos_index = []
neg_index = []
neut_index = []
#This loop saves the indexs of rows of dataframe which correspond to specific tweets by classes
for i in range(len(train_features)):
    if train_labels[0][i] == "positive":
        pos_index.append(i)
    elif train_labels[0][i] == "negative":
        neg_index.append(i)
    else:
        neut_index.append(i)
        
#calculating prior probabilities by class
prior_pos = len(pos_index)/(len(neg_index)+len(pos_index)+len(neut_index))
prior_neg = len(neg_index)/(len(neg_index)+len(pos_index)+len(neut_index))
prior_neut = len(neut_index)/(len(neg_index)+len(pos_index)+len(neut_index))

#This function returns the number of total words in trainingset for certain class
def total_words_by_class(class_index):
    sum_class = []
    for j in range(len(class_index)):
        sum_class.append(train_features.transpose()[class_index[j]].sum())
    return sum(sum_class)
    
total_neut_words = total_words_by_class(neut_index)
total_pos_words = total_words_by_class(pos_index)
total_neg_words = total_words_by_class(neg_index)

#This function returns the likelihoods for multinomial case.
#For Drichlet smoothing, set alpha to 1, otherwise set it to 0
#M stands for multinomial case
def M_likelihood_by_class(class_index, total_class_words, alpha):
    temp = 0
    likelihoods =[]
    for m in range(5722):
        for n in range(len(class_index)):
            temp = temp + train_features[m][class_index[n]]
        likelihoods.append(temp + alpha)
        temp = 0
    return likelihoods/(total_class_words + alpha*5722) 

#Calculating thetas for all classes i.e neutral, positive and negative
theta_mle_neut_M = M_likelihood_by_class(neut_index, total_neut_words,0)
theta_mle_pos_M = M_likelihood_by_class(pos_index, total_pos_words,0)
theta_mle_neg_M = M_likelihood_by_class(neg_index, total_neg_words,0)

#Definition of log funtion to include valid answer for log(0) i.e. negative infinity
def log_10 (num):
    if num != 0:
        return math.log10(num)
    else:
        return -math.inf
#This function calculates posterior probabilities for specific class using likelihoods and priors for Multinomial case
def M_posterior_by_class(M_theta_mle_by_class, prior_class):
    post_class = 0
    posteriors = []
    for s in range(2928):
        for t in range(5722):
            if ((test_features[t][s] != 0) or (M_theta_mle_by_class[t] != 0)):
                post_class = post_class + test_features[t][s]*log_10(M_theta_mle_by_class[t])
        posteriors.append(post_class + math.log10(prior_class))
        post_class = 0
    return posteriors 

#Calculating posterior probabilities for all classes i.e neutral, positive and negative
posterior_neut_M = M_posterior_by_class(theta_mle_neut_M, prior_neut)
posterior_pos_M = M_posterior_by_class(theta_mle_pos_M, prior_pos)
posterior_neg_M = M_posterior_by_class(theta_mle_neg_M, prior_neg)    

#This function is used for prediction of new tweets (test_set) and returns accuracy and number of wrong predictions 
def prediction_and_accuracy(posterior_neut,posterior_pos, posterior_neg):
    predictions = []
    true_predict_count = 0
    
    for y in range(2928):
        m = max(posterior_neut[y],posterior_pos[y],posterior_neg[y])
        if (m == posterior_neut[y]):
            predictions.append("neutral")
        elif (m == posterior_neg[y]):
            predictions.append("negative")
        else:
            predictions.append("positive")     
            
    for z in range(2928):
        if predictions[z] == test_labels[0][z]:
            true_predict_count = true_predict_count + 1
    
    wrong_predictions = 2928 - true_predict_count      
    accuracy = (true_predict_count/2928)*100
    return accuracy, wrong_predictions
#Calculating accuracy and number of wrong preductions for multinomial case on test set
Multinomial_Results = prediction_and_accuracy(posterior_neut_M,posterior_pos_M, posterior_neg_M)
print(Multinomial_Results)

#Drichlet Smoothing from now onwards

#Calculating thetas for all classes i.e neutral, positive and negative in case of Smoothing
smooth_theta_mle_neut = M_likelihood_by_class(neut_index, total_neut_words,1)
smooth_theta_mle_pos = M_likelihood_by_class(pos_index, total_pos_words,1)
smooth_theta_mle_neg = M_likelihood_by_class(neg_index, total_neg_words,1)

#Calculating posterior probabilities for all classes i.e neutral, positive and negative in case of Smoothing
smooth_posterior_neut = M_posterior_by_class(smooth_theta_mle_neut, prior_neut)
smooth_posterior_pos = M_posterior_by_class(smooth_theta_mle_pos, prior_pos)
smooth_posterior_neg = M_posterior_by_class(smooth_theta_mle_neg, prior_neg)

#Calculating accuracy and number of wrong preductions for Drichlet Smoothing case on test set (Multinomial)
Drichlet_Results = prediction_and_accuracy(smooth_posterior_neut,smooth_posterior_pos, smooth_posterior_neg)
print(Drichlet_Results)

#Bernoulli part from now onwards 

#This function returns the likelihoods for Bernoulli case.
#B stands for Bernoulli case
def B_likelihood_by_class(class_index):
    temp = 0
    likelihoodss =[]
    for m in range(5722):
        for n in range(len(class_index)):
            if train_features[m][class_index[n]] != 0:
                temp = temp + 1
        likelihoodss.append(temp )
        temp = 0
    return likelihoodss/(np.int64(len(class_index))) 

#Calculating thetas for all classes i.e neutral, positive and negative for Bernoulli case
theta_mle_neut_B = B_likelihood_by_class(neut_index)
theta_mle_pos_B = B_likelihood_by_class(pos_index)
theta_mle_neg_B = B_likelihood_by_class(neg_index)

#This function calculates posterior probabilities for specific class using likelihoods and priors for Bernoulli case    
def posterior_by_class(theta_mle_by_class, prior_by_class):
    post_class = 0
    posteriorss = []
    for s in range(2928):
        for t in range(5722):
            if ((test_features[t][s] != 0)):
                post_class = post_class + log_10(theta_mle_by_class[t])
            else:
                post_class = post_class + log_10( 1 - theta_mle_by_class[t])
        posteriorss.append(post_class + math.log10(prior_by_class))
        post_class = 0
    return posteriorss 

#Calculating posterior probabilities for all classes i.e neutral, positive and negative for Bernoulli case
posterior_neut_B = posterior_by_class(theta_mle_neut_B, prior_neut)
posterior_pos_B = posterior_by_class(theta_mle_pos_B, prior_pos)
posterior_neg_B = posterior_by_class(theta_mle_neg_B, prior_neg)  
#Calculating accuracy and number of wrong preductions for Bernoulli case on test set
Bernoulli_Results = prediction_and_accuracy(posterior_neut_B,posterior_pos_B, posterior_neg_B)
print(Bernoulli_Results)

#Last Part from now onwards to find most common 20 words by class

#This function finds the most common 20 words according to given class
def common_20(class_index):
    count = 0
    word_occurences =[]
    common_words_20 = []
    for m in range(5722):
        for n in range(len(class_index)):
            count = count + train_features[m][class_index[n]]
        word_occurences.append(count)
        count = 0
    word_occurences_sorted = np.flip(np.argsort(word_occurences))
    for i in range(20):
        common_words_20.append(vocab[0][word_occurences_sorted[i]])
    return common_words_20
    
#calculating most common 20 words for each class
common_words_20_pos = common_20(pos_index)
print(common_words_20_pos)
common_words_20_neg = common_20(neg_index)
print(common_words_20_neg)
common_words_20_neut = common_20(neut_index)
print(common_words_20_neut)
