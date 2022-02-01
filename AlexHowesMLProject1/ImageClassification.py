from cProfile import label
from calendar import c
import math
from re import S
from matplotlib import pyplot as plt
from mxnet.gluon.data.vision import MNIST
import numpy as np
p_on = 1
#laplace smoothing
#??? is this 28^2 or 2? i think 28^2 but thats too high
vocab_size = 28*28
#list of 2d matrices for keeping track of pixel frequencies
accuracy = []
k_trained = []
accuracy_by_digit = []
def parse_labels(train_x,train_labels, labels):
    label_count = []
    label_dict = {}
    total_sum = 0
    for current_digit in labels:
        digit_indices = np.where(train_labels == current_digit)
        label_dict[current_digit] = np.array(train_x[digit_indices])
        total_sum_of_digit = len(digit_indices[0])
        label_count.append(total_sum_of_digit)
        total_sum += total_sum_of_digit
    prior_probabilities = [label / total_sum for label in label_count]
    log_prior = [math.log(label/float(total_sum)) for label in label_count]
    return log_prior, label_dict

def conditional_probability(word, total_words_in_event,k):
        return (word + k) / float((total_words_in_event + k*vocab_size))

def test_model_image(pixel_frequencies, test_x, test_y,k,log_prior,labels):
    predicted_labels = []
    #test_x = t
    #test_y = np.array([0,1,2,3,4])
    
    i = 0
    correct_predictions = 0
    c_p_digit = np.zeros(shape=(10))
    c_t_digit = np.zeros(shape=(10))
    sum_of_frequency = []
    print(len(test_x))
    i = 0
    for image in test_x:
        #for each on pixel value grab the value that exists in pixel frequencies
        #then calculate the conditional probabiility using log to avoid underflow
        r,c = np.where(image >= p_on)
        probabilities = [prior for prior in log_prior]
        #print(probabilities)
        for digit in labels:
            #list of frequencies for each on pixel
            freq = pixel_frequencies[digit][r,c]
            for f in freq:
                probabilities[digit] = probabilities[digit] + math.log(conditional_probability(f, freq.sum(),k))
            #print(f"probability for digit {digit} is {probabilities[digit]}")
        prediction = np.argmax(probabilities)
        #print(f"ACTUAL DIGIT: {test_y[i]} | PREDICTED {prediction}")
        if(test_y[i] == prediction): 
            correct_predictions += 1
            c_p_digit[test_y[i]] +=1
        c_t_digit[test_y[i]] +=1
        i+=1
    print(f"accuracy k {k} : {correct_predictions} / {len(test_y)} = {correct_predictions/float(len(test_y))}")
    accuracy.append((correct_predictions/(len(test_y))))
    k_trained.append(k)
    digits = []
    for digit in range(10):
        print(f"accuracy by digit {digit} : {c_p_digit[digit]} / {c_t_digit[digit]} = {c_p_digit[digit]/float(c_t_digit[digit])}")
        digits.append(c_p_digit[digit]/float(c_t_digit[digit]))
    accuracy_by_digit.append(digits)




X = MNIST(train=True)
x,y = X[:]
x = np.squeeze(x.asnumpy(), axis=3)
#calculate prior probabilities from samples
#i.e. num of certain digit / total samples
labels = list(range(0, 10))

k=0.5
folds = 5
partition = math.floor((len(x)/folds))
partitions = [partition * i for i in range(folds)]
partitions.append(len(x))
for i in range(folds):
    test_x = x[partitions[i]:partitions[i+1]]
    test_labels = y[partitions[i]:partitions[i+1]]
    train_x = x.copy()
    train_labels = y.copy()
    log_prior, label_dict = parse_labels(train_x, train_labels, labels)    
    #calculate the individual probabilities for each pixel
    #use frequency for multinomial
    #PARAMETERS
    pixel_frequencies=np.zeros((10,28,28))
    #pixel_frequencies = np.zeros((10,5,5))
    for current_digit in labels:
        #calculate the frequency for each pixel within each image of the specified digit
        #this will grab the locations of pixels where it exceeds the intesity threshold
        for image in label_dict[current_digit]:
            r,c = np.where(image >= p_on)
            #location = np.vstack((r,c)).T
            #add one to each location containing an on pixel in the pixel frequencies
            np.add.at(pixel_frequencies[current_digit], [r,c], 1)
    test_model_image(pixel_frequencies, test_x, test_labels,k, log_prior,labels)
    k+=0.5
#now we have the frequency that each pixel is on for each digit
#we can calculate conditional probabilities now with new images!
#say pixel 4,5 is ON P(7) = (prior of 7) * p((4,5) ON | 7)
#p((4,5) ON | 7) = amount of times that pixel 4,5 is on for 7 over the total 7 samples (with smoothing)
#print(f"pixel frequencies {pixel_frequencies}")
best_acc_index = np.argmax(np.array(accuracy))
print(best_acc_index, k_trained[best_acc_index])
test = MNIST(train=False)
test_x, test_y = test[2:300]
test_x = np.squeeze(test_x.asnumpy(), axis=3)
test_model_image(pixel_frequencies, test_x, test_y, k_trained[best_acc_index],log_prior,labels)


print(accuracy_by_digit)
print(accuracy)
print(k_trained)
plt.plot(k_trained, accuracy)
plt.xlabel("k (strength factor)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs K Smoothing Factor (Multinomial Image Classification)")
plt.show()
plt.plot(k_trained, accuracy_by_digit)
plt.xlabel("k(strength factor)")
plt.ylabel("Accuracy (%)")
plt.title("Individual Digit Accuracy vs K Smoothing Factor (Multinomial Image Classification)")
plt.legend([0,1,2,3,4,5,6,7,8,9])
plt.show()
