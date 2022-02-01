from csv import DictReader
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import matplotlib.pyplot as plt
accuracy = []
k_tested = []
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

def calc_cond_prob(nk, total_event_words,k):
    return (nk + k) / float((total_event_words + k*len(spam_sum)))

def test_model(test_msgs, test_labels,k):
    print(f"k = {k}")
    i = 0
    correct_spam = 0
    t_spam = 0
    correct_ham = 0
    t_ham = 0
    total_pred = 0
    for msg in test_msgs:
        prob_spam = math.log(p_spam)
        prob_ham = math.log(p_ham)
        for word in msg.lower().split():
            #calculate the probability that the msg is spam based on the word
            try:
                nk = spam_sum[word]
            except KeyError:
                nk = 0
            prob_spam = prob_spam + math.log(calc_cond_prob(nk, total_spam_words,k))
            try:
                nk = ham_sum[word]
            except KeyError:
                nk = 0
            prob_ham = prob_ham + math.log(calc_cond_prob(nk, total_ham_words,k))
        #print(f"prob_spam {prob_spam} | prob_ham {prob_ham}")
        prediction = np.argmax([prob_spam, prob_ham])
        #print(f"Prediction {prediction} | Nominal {test_labels[i]}")
        if label_names[prediction] == test_labels[i]:
            if prediction == 0: correct_spam +=1
            else: correct_ham += 1
        if test_labels[i] == label_names[0]:
            t_spam+=1
        else:
            t_ham+=1
        total_pred+=1
        i+=1
    print(f"accuracy: {correct_spam + correct_ham} / {total_pred} = {(correct_ham+correct_spam)/total_pred}")
    print(f"accuracy for ham: {correct_ham} / {t_ham} = {correct_ham/t_ham}")
    print(f"accuracy for spam: {correct_spam} / {t_spam} = {correct_spam/t_spam}")
    acc = (correct_spam+correct_ham)/total_pred
    accuracy.append(acc)
    k_tested.append(k)
    ham_accuracy.append(correct_ham/t_ham)
    spam_accuracy.append(correct_spam/t_spam)



multinom_cv = CountVectorizer(analyzer = "word", 
                             lowercase=True, 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = {'english'},
                             token_pattern='[a-zA-Z0-9$&+:;=@#|<>^*()%-]+')

messages, labels = retrieve_SMS_data(training=True)
folds = 5

partition = math.floor((len(messages)/folds))
partitions = [partition * i for i in range(folds)]
partitions.append(len(messages))
k = 0.5
for i in range(folds):
    validation_test = messages[partitions[i]:partitions[i+1]]
    validation_labels = labels[partitions[i]:partitions[i+1]]
    msgs = messages.copy()
    lab = labels.copy()
    del msgs[partitions[i]:partitions[i+1]]
    del lab[partitions[i]:partitions[i+1]]
    x = multinom_cv.fit_transform(msgs)
    vocabulary = multinom_cv.vocabulary_
    features = multinom_cv.get_feature_names_out()
    msg_list = x.toarray()
    #utilize Pandas to map messages to their respective labels
    dataFrame = pd.DataFrame(data=msg_list, columns=features)
    label_names = ["Spam","Non-Spam"]
    dataFrame["labels"] = lab
    spam_rows = dataFrame[dataFrame["labels"] == label_names[0]]
    ham_rows = dataFrame[dataFrame["labels"] == label_names[1]]
    del spam_rows["labels"]
    del ham_rows["labels"]
    spam_sum = spam_rows.sum(axis=0,skipna=True)
    ham_sum = ham_rows.sum(axis=0,skipna=True)
    total_spam_words = spam_rows.to_numpy().sum()
    total_ham_words = ham_rows.to_numpy().sum()
    # we have all of the items we need to test the multinomial classifier
    p_spam = lab.count(label_names[0]) / float(len(lab))
    p_ham = lab.count(label_names[1]) / float(len(lab))
    test_model(validation_test,validation_labels,k)
    k+=0.5


test_msgs, test_labels = retrieve_SMS_data(False)
best_acc_index = np.argmax(np.array(accuracy))
print(best_acc_index, k_tested[best_acc_index])
test_model(test_msgs,test_labels,k_tested[best_acc_index])

#MULTINOMIAL
#for each word found in the message we calculate the probability that it is spam and ham


print(accuracy)
print(k_tested)
plt.plot(k_tested, accuracy)
plt.xlabel("k (strength factor)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs K Smoothing Factor (Multinomial Text Classification)")
plt.show()
plt.plot(k_tested, spam_accuracy)
plt.xlabel("k (strength factor)")
plt.ylabel("Spam Accuracy (%)")
plt.title("Spam Accuracy vs K Smoothing Factor (Multinomial Text Classification)")
plt.show()
plt.plot(k_tested, ham_accuracy)
plt.xlabel("k (strength factor)")
plt.ylabel("Ham Accuracy (%)")
plt.title("Ham Accuracy vs K Smoothing Factor (Multinomial Text Classification)")
plt.show()