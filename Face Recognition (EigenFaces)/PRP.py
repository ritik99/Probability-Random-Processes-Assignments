'''
Author: Ritik Dutta
Roll No.: 16110053
'''


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import linalg as LA
np.set_printoptions(threshold=np.inf, precision = 3)
from mpl_toolkits import mplot3d
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

dir = './orl_faces/'
non_face_dir = './cars_brad'
outside_faces_dir = './faces'

nonface_ = []
img_names = os.listdir(non_face_dir + '/')
img_count = 0

#############################################################################
#############################################################################
#############################################################################

#The following part is used to load the three image datasets onto the program

for i in img_names:
    if img_count < 6:
        t = cv.imread(non_face_dir + '/' + i)
        t = cv.cvtColor(t, cv.COLOR_BGR2GRAY)
        t = cv.resize(t, (112, 92))
        t = t.reshape(1, t.shape[0] * t.shape[1])
        nonface_.append(t)
        img_count += 1
    else:
        break
nonface_np = np.array(nonface_[0])
for i in range(1, len(nonface_)):
    nonface_np = np.vstack((nonface_np, nonface_[i]))

outsidefaces_ = []
img_names = os.listdir(outside_faces_dir + '/')
img_count = 0
for i in img_names:
    try:
        if img_count < 6:
            t = cv.imread(outside_faces_dir + '/' + i)
            #print(t.shape)
            t = cv.cvtColor(t, cv.COLOR_BGR2GRAY)
            t = cv.resize(t, (112, 92))
            t = t.reshape(1, t.shape[0] * t.shape[1])
            outsidefaces_.append(t)
            img_count += 1
    except:
        break

outsidefaces_np = np.array(outsidefaces_[0])
for i in range(1, len(outsidefaces_)):
    outsidefaces_np = np.vstack((outsidefaces_np, outsidefaces_[i]))


folders = []

for file in os.listdir(dir):
    if os.path.isdir(dir + file):
        folders.append(file)

#print(folders)
X = []
cls = []
og_shape = (0, 0)
folder_num = 0
for folder in folders:
    path = dir + folder
    img_names = os.listdir(path + '/')
    temp = []
    temp2 = []
    for i in img_names:
        #print('here')
        t = cv.imread(path + '/' + i)
        t = cv.cvtColor(t, cv.COLOR_BGR2GRAY)
        og_shape = t.shape
        #print(og_shape)
        t = t.reshape(1, t.shape[0] * t.shape[1])
        temp.append(t)
        temp2.append(folder_num)
    cls.append(temp2)
    folder_num += 1
    X.append(temp)
X_train_list = []
Y_train_list = []
X_test_list = []
Y_test_list = []
for i in range(len(X)):
    temp1, temp2, temp3, temp4 = train_test_split(X[i], cls[i], test_size=0.2)
    X_train_list.append(temp1)
    X_test_list.append(temp2)
    Y_train_list.append(temp3)
    Y_test_list.append(temp4)


#############################################################################
#############################################################################
#############################################################################

#Next, the mean vector is calculated


mean_vec = np.zeros(X[0][0].shape)
num_img = 0

for i in range(len(X_train_list)):
    for j in range(len(X_train_list[i])):
        mean_vec += X_train_list[i][j]
        num_img += 1

mean_vec = mean_vec / num_img

X_train = []
X_test = []
Y_train = []
Y_test = []

for i in range(len(X_train_list)):
    temp = []
    for j in range(len(X_train_list[i])):
        temp.append(X_train_list[i][j] - mean_vec)   #Mean vector is sutracted from all the input training images
    X_train.append(temp)


X_test.append(X_test_list[0][0])
Y_test.append(Y_test_list[0][0])
Y_train.append(Y_train_list[0][0])

X_train = np.array(X_train[0][0])
X_test = np.array(X_test[0][0])
Y_train = np.array(Y_train[0])
Y_test = np.array(Y_test[0])

for i in range(len(X_train_list)):
    for j in range(len(X_train_list[i])):
        if i != 0 or j != 0:
            X_train = np.vstack((X_train, X_train_list[i][j]))
            Y_train = np.vstack((Y_train, Y_train_list[i][j]))

for i in range(len(X_test_list)):
    for j in range(len(X_test_list[i])):
        if i != 0 or j != 0:
            X_test = np.vstack((X_test, X_test_list[i][j]))
            Y_test = np.vstack((Y_test, Y_test_list[i][j]))


#############################################################################
#############################################################################
#############################################################################

#Here, the algorithm to recognise faces using eigenvectors is implemented

YY = np.dot(X_train, np.transpose(X_train))
eigVal, eigVec = LA.eig(YY)
eigValdict = {}
pos = 0
for i in eigVal:
    eigValdict[pos] = i
    pos += 1

argSorteig = np.argsort(eigVal)

sortedeigVal = np.zeros(eigVal.shape)
sortedeigVec = np.zeros(eigVec.shape)

#print(sortedeigVec.shape)
for i in range(eigVal.shape[0]):
    sortedeigVal[i] = eigVal[argSorteig[i]]
    #print(eigVec[:, argSorteig[i]].shape)
    sortedeigVec[:, i] = eigVec[:, argSorteig[i]]

neweigVec = np.dot(np.transpose(X_train), sortedeigVec)
neweigVec = sklearn.preprocessing.normalize(neweigVec, axis=0)


sample = neweigVec[:, neweigVec.shape[1] - 5].reshape(og_shape)
sampleT = np.transpose(sample)
basis = neweigVec[:, neweigVec.shape[1] - 10 : neweigVec.shape[1] - 1]


#This part displays the 9 eigenfaces being used as the basis. Uncomment it to display the eigenfaces
'''for i in range(basis.shape[1]):
    temp = basis[:, i].reshape(112, 92)
    plt.imshow(temp, cmap='gray')
    plt.show()'''
basis = np.transpose(basis)


#This function calculates the weights

def weights(X):
    temp = np.dot(basis, np.transpose(X))
    return temp


#This function recognises the faces in the test set

acc = 0
for i in range(X_test.shape[0]):
    dist = 10000000000
    pID = -1
    for j in range(X_train.shape[0]):
        #temp = X_test[i, :] - mean_vec
        norm = np.linalg.norm(weights(X_test[i, :]) - weights(X_train[j, :]))
        if norm < dist:
            #print('j ', Y_train[j])
            #print('here')
            pID = Y_train[j]
            dist = norm
    if pID == Y_test[i]:
        acc += 1
    print('Predicted person: ', pID, 'Real category: ', Y_test[i], ' Distance: ', dist)
print('accuracy : ', acc/X_test.shape[0])

#This function recognises non-face backgroun images

for i in range(nonface_np.shape[0]):
    dist = 100000000000
    pID = -1
    for j in range(X_train.shape[0]):
        norm = np.linalg.norm(weights(nonface_np[i, :]) - weights(X_train[j, :]))
        if norm < dist:
            # print('j ', Y_train[j])
            # print('here')
            dist = norm
    print('Not a face. Distance: ', dist)

#This function detects the faces not in the original dataset

for i in range(outsidefaces_np.shape[0]):
    dist = 100000000000
    pID = -1
    for j in range(X_train.shape[0]):
        norm = np.linalg.norm(weights(outsidefaces_np[i, :]) - weights(X_train[j, :]))
        if norm < dist:
            # print('j ', Y_train[j])
            # print('here')
            dist = norm
    print('Face detected. Distance: ', dist)

'''face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
for i in range(outsidefaces_np.shape[0]):
    t = outsidefaces_np[i].reshape(112, 92)
    print(t)
    faces = face_cascade.detectMultiScale(t, 1.3, 5)
    print(faces)

for i in range(nonface_np.shape[0]):
    t = nonface_np[i].reshape(112, 92)
    faces = face_cascade.detectMultiScale(t, 1.3, 5)
    print(faces)'''





