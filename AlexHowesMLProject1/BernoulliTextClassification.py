from csv import DictReader
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math

train_accuracies = []
k_train = []
spam_accuracy = []
ham_accuracy = []

def retrieve_SMS_data(training):
    if training == True:
        file_path = "spha\\SMS_train.csv"
    else:
        file_path = "spha\\SMS_test.csv"    
    data = []
    labels = []
    with open(file_path, 'r', encoding="utf-8") as dataset_reader:
        dict_reader = DictReader(dataset_reader)
        for row in dict_reader:            
            data.append(row.get("Message_body"))
            labels.append(row.get("Label"))
    return data, labels


def calc_cond_prob(nk, total_event_words, vocab_size, k):
    return (nk + k) / float((total_event_words + k*vocab_size))

def test_model(test_msgs, test_labels, total_spam_email, total_ham_email, spam_sum, ham_sum, p_spam_log, p_ham_log,k):
    i=0
    correct_prediction = 0
    correct_spam = 0
    correct_ham = 0
    t_s = 0
    t_h = 0
    label_names = ["Spam", "Non-Spam"]
    for msg in test_msgs:
        msg_words = msg.lower()
        prob_spam = p_spam_log
        prob_ham = p_ham_log
        #print(f"MESSAGE:  {msg_words}")
        for word in vocabulary:
            #check to see if word exists in message
            in_word = False
            try:
                nk = spam_sum[word]
            except KeyError:
                nk = 0
            try:
                nk_ham = ham_sum[word]
            except KeyError: 
                nk_ham = 0
            if word in msg_words:
                prob_spam = prob_spam + math.log(calc_cond_prob(nk, total_spam_email, len(spam_sum), k))
                #calculate probability of ham based on the fact that the word exists in email
                prob_ham = prob_ham + math.log(calc_cond_prob(nk_ham, total_ham_email, len(spam_sum), k))
            else:
                #if word is not in the message calculate how many spails do not have the word
                spam_without_word = total_spam_words - nk            
                prob_spam = prob_spam + math.log(calc_cond_prob(spam_without_word, total_spam_email, len(spam_sum), k))
                ham_without_word = total_ham_words - nk_ham
                prob_ham = prob_ham + math.log(calc_cond_prob(ham_without_word, total_ham_email, len(spam_sum), k))
            #print(f"in word {in_word} | prob_ham {prob_ham} | prob_spam{prob_spam}")
        #print(f"prob spam {prob_spam} | prob ham {prob_ham}")
        prediction = np.argmax([prob_spam,prob_ham])
        #print(f"PREDICTION {label_names[prediction]} | NOMINAL {test_labels[i]}")
        if(label_names[prediction] == test_labels[i]):
            correct_prediction +=1
            if(test_labels[i] == label_names[0]):
                correct_spam+=1
            else:
                correct_ham +=1
        if(test_labels[i] == label_names[0]):
            t_s+=1
        else:
            t_h+=1
                
        i+=1
    print(f"accuracy {correct_prediction/len(test_labels)}")
    train_accuracies.append(correct_prediction/len(test_labels))
    print(f"spam correct: {correct_spam} / {t_s} = {correct_spam/t_s}")
    spam_accuracy.append(correct_spam/t_s)
    print(f"ham correct: {correct_ham} / {t_h} = {correct_ham/t_h}")
    ham_accuracy.append(correct_ham/t_h)
    k_train.append(k)


messages, labels = retrieve_SMS_data(training=True)
multinom_cv = CountVectorizer(analyzer = "word", 
                             lowercase=True, 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = {'english'}, 
                             token_pattern='[a-zA-Z0-9$&]+',
                             binary=True)

folds = 5
#5 fold cross validation
partition = math.floor((len(messages))/folds)
partitions = partitions = [partition * i for i in range(folds)]
partitions.append(len(messages))
k=0.5
for i in range(folds):
    #HAVE TO DO SEPERATE VECTORIZE WITH BINARY SET TO TRUE
    validation_test = messages[partitions[i]:partitions[i+1]]
    validation_labels = labels[partitions[i]:partitions[i+1]]
    print(f"partitions: {partitions[i]} -> {partitions[i+1]}")
    train_msgs = messages.copy()
    del train_msgs[partitions[i]:partitions[i+1]]
    train_labels = labels.copy()
    del train_labels[partitions[i]:partitions[i+1]]
    print(len(labels) , print(len(messages)))
    x = multinom_cv.fit_transform(train_msgs)
    vocabulary = multinom_cv.vocabulary_
    features = multinom_cv.get_feature_names_out()
    msg_list = x.toarray()

    dataFrame = pd.DataFrame(data=msg_list, columns=features)
    label_names = ["Spam", "Non-Spam"]
    #dataFrame["labels"] = labels
    dataFrame["labels"] = train_labels
    spam_rows = dataFrame[dataFrame["labels"] == label_names[0]]
    ham_rows = dataFrame[dataFrame["labels"] == label_names[1]]
    del spam_rows["labels"]
    del ham_rows["labels"]
    spam_sum = spam_rows.sum(axis=0, skipna=True)
    ham_sum = ham_rows.sum(axis=0, skipna=True)
    total_spam_words = spam_rows.to_numpy().sum()
    total_ham_words = ham_rows.to_numpy().sum()
    total_spam_email = labels.count(label_names[0])
    total_ham_email = labels.count(label_names[1])
    p_spam = total_spam_email / float(len(train_labels))
    p_ham = total_ham_email / float(len(train_labels))
    #test validation
    p_spam_log = math.log(p_spam)
    p_ham_log = math.log(p_ham)
    test_model(validation_test, validation_labels, total_spam_email, total_ham_email, spam_sum, ham_sum, p_spam_log, p_ham_log, k)
    k+=0.5



best_acc_index = np.argmax(np.array(train_accuracies))
print(best_acc_index, k_train[best_acc_index])

test_msgs, test_labels = retrieve_SMS_data(False)
test_model(test_msgs,test_labels,total_spam_email,total_ham_email,spam_sum, ham_sum, p_spam_log, p_ham_log,k_train[best_acc_index])
print(train_accuracies)
print(k_train)
print(spam_accuracy)
print(ham_accuracy)
plt.plot(k_train, train_accuracies)
plt.xlabel("k (strength factor)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K Smoothing Factor (Bernoulli Text Classification")
plt.show()
plt.plot(k_train, spam_accuracy)
plt.xlabel("k (strength factor)")
plt.ylabel("Accuracy for Spam Emails")
plt.title("Spam Accuracy vs K Smoothing Factor (Bernoulli Text Classification")
plt.show()
plt.plot(k_train, ham_accuracy)
plt.xlabel("k (strength factor)")
plt.ylabel("Accuracy for Ham Emails")
plt.title("Ham Accuracy vs K Smoothing Factor (Bernoulli Text Classification")
plt.show()
#BERNOULLI
#for each word in the vocabulary we calculate the probability if the message is spam based
#on if the word is present or not
#p(spam) = #number of emails with word that is spam + k / num of spam + k*|vocab|
#p(spam) (not present) = number of emails without word that is spam + k / num of spam + k*|vocab|



         