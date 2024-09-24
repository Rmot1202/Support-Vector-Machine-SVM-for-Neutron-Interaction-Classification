# reserve 10% of the data for our own testing after the fact
#X_train, X_test, y_train, y_test = train_test_split(input_data_min_max, label, test_size=0.1, stratify=label, random_state=1);
X = pd.DataFrame(input_data_min_max).iloc[:,[3,4,5,6,7,9,10,11,12,13]]
y = pd.DataFrame(label).iloc[:,:]
scaler = StandardScaler()
Dataset_featured = scaler.fit_transform(X)
print(Dataset_featured)
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(Dataset_featured, y, test_size = 0.4, random_state =3)
# Printing shapes of training and testing sets
'''
print("Training set - Features shape:", X_train.shape)
print("Training set - Labels shape:", y_train.shape)
print("Testing set - Features shape:", X_test.shape)
print("Testing set - Labels shape:", y_test.shape)

# Optionally, you can print the actual data if needed
print("Training set - Features:")
print(X_train)
print("Training set - Labels:")
print(y_train)
print("Testing set - Features:")
print(X_test)
print("Testing set - Labels:")
print(y_test)
'''
'''XA = dataset.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13]]
XB= dataset.iloc[:,[2,3,6,7,8,9,12,13]]
scaler = StandardScaler()
Dataset_featuredA = scaler.fit_transform(XA)
Dataset_featuredB = scaler.fit_transform(XB)
X_trainA, X_testA, y_trainA, y_testA= train_test_split(Dataset_featuredA, y, test_size = 0.4, random_state =3)
X_trainB, X_testB, y_trainB, y_testB= train_test_split(Dataset_featuredB, y, test_size = 0.4, random_state =3)'''

U = 1
num=[0]*100
accuracyA=[0]*100
#accuracyB=[0]*100
#accuracyC=[0]*100
while U != 100:  # Added colon to the while loop
    clfA = svm.SVC(kernel='linear', C=U, gamma='auto')   #Corrected kernel indexing
    #clfB = svm.SVC(kernel='linear', C=U, gamma='auto')
    #clfC = svm.SVC(kernel='linear', C=U, gamma='auto')
    clfA.fit(X_train, y_train)  # Assuming X_train and y_train are defined
    #clfB.fit(X_trainA, y_trainA)
    #clfC.fit(X_trainB, y_trainB)
    y_pred = clfA.predict(X_test)
    #y_predA = clfB.predict(X_testA)
    #y_predB = clfC.predict(X_testB)
    accuracy = accuracy_score(y_test, y_pred)
    #accuracy1 = accuracy_score(y_testA, y_predA)
    #accuracy2 = accuracy_score(y_testB, y_predB)
    accuracyA[U]=accuracy
    #accuracyB[U]=accuracy1
    #accuracyC[U]=accuracy2
    num[U]=U
    print(  f"{U}, Accuracy: {accuracy}")
    U += 1# Increment U to progress through the kernels
    '''
plt.plot(num,accuracyA,label="x and y")
plt.plot(num,accuracyB,label="all" )
plt.plot(num,accuracyC,label="BarID" )
# naming the x axis
plt.xlabel('U')
# naming the y axis
plt.ylabel('Accuracy')
 
# giving a title to my graph
plt.title('Affect of U on accuracy gamma=auto')
 
plt.legend()'''



    
'''print(y_pred)
print(y_test.values)'''

'''U = 1
num = [0] * 100
accuracy1 = [0] * 100
accuracy2 = [0] * 100
accuracy3 = [0] * 100
while U != 100:  # Added colon to the while loop
    clf = svm.SVC(kernel='linear', C=U,gamma='auto')  # Corrected kernel indexing
    clf1 = svm.SVC(kernel='linear', C=U, gamma='scale')
    clf2 = svm.SVC(kernel='linear', C=U, gamma=U)
    clf.fit(X_train, y_train)  # Assuming X_train and y_train are defined
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred1 = clf1.predict(X_test)
    y_pred2 = clf2.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracya = accuracy_score(y_test, y_pred1)
    accuracyb = accuracy_score(y_test, y_pred2)
    accuracy1[U] = accuracy
    accuracy2[U] = accuracya
    accuracy3[U] = accuracyb
    num[U] = U
    print(  f"{U}, Accuracy: {accuracy}")
    print(  f"{U}, Accuracy: {accuracya}")
    print(  f"{U}, Accuracy: {accuracyb}")
    U += 1  # Increment U to progress through the kernels
plt.plot(num, accuracy1, label="gamma=auto")
plt.plot(num, accuracy2, label="gamma= Scale")
plt.plot(num, accuracy3, label="Gamma=C")

# naming the x axis
plt.xlabel('U')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
plt.title('Affect of gamma on accuracy')

plt.legend()'''

#clf = svm.SVC(kernel='linear', C=19)  # Corrected kernel indexing
#clf.fit(X_train, y_train)  # Assuming X_train and y_train are defined
 '''Input values
t0 = float(input("Enter t0: "))
x0 = float(input("Enter x0: "))
y0 = float(input("Enter y0: "))
z0 = float(input("Enter z0: "))
q0 = float(input("Enter q0: "))
t1 = float(input("Enter t1: "))
x1 = float(input("Enter x1: "))
y1 = float(input("Enter y1: "))
z1 = float(input("Enter z1: "))
q1 = float(input("Enter q1: "))

# Create an array with the inputs
inputs = [[t0, x0, y0, z0, q0, t1, x1, y1, z1, q1]]

# Initialize a scaler and scale the inputs
inputs_scaled = scaler.fit_transform(inputs)

# Initialize the classifier
clf = SVC()

# Make prediction
y_pred = clf.predict(inputs_scaled)
print("Predicted class:", y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score
import time
# load ROOT
import ROOT
# load TNeutron class
ROOT.gSystem.Load('/home/user/data/ML/lib/v6_30/TNeutron_cc.so')
# set the path to our simulation file
#datapath = "/home/user/data/ML/MLP_mult2/"
#datapath = "/home/user/data/ML/e15118/"
datapath = "/home/user/data/ML/stmona/"
# open the ROOT file and load the TTree that it contains
#rootfile = ROOT.TFile(datapath + "delta18_O262plus_2nThermal2.root","READ")
# try this one for training
rootfile = ROOT.TFile(datapath + "tneutron_mult2_26O_24O+2n-uniform.root","READ")
# try this one to test your trained model
rootfile1 = ROOT.TFile(datapath + "tneutron_mult2_26O_24O+2n-delta500keV.root","READ")
tree1= rootfile1.Get("snt")
tree = rootfile.Get("snt")
#tree.Print()
# copy data from the tree
entries = tree.GetEntries();
hit_data = np.zeros((entries, 10));
label = np.empty(entries);
label.fill(9);
for i in range(entries):
  # get the tree index for the next entry on our list
  # and load this entry from the TTree
  tree.GetEntry(i)
  # transfer the data for this entry to our numpy array
  # x0
  hit_data[i][0] = tree.g.x[0]
  # y0
  hit_data[i][1] = tree.g.y[0]
  # z0
  hit_data[i][2] = tree.g.z[0]
  # t0
  hit_data[i][3] = tree.g.t[0]
  # q0
  hit_data[i][4] = tree.g.q[0]
  # x1
  hit_data[i][5] = tree.g.x[1]
  # y1
  hit_data[i][6] = tree.g.y[1]
  # z1
  hit_data[i][7] = tree.g.z[1]
  # t1
  hit_data[i][8] = tree.g.t[1]
  # q1
  hit_data[i][9] = tree.g.q[1]
  # label
  label[i]= tree.signal
  #label[i]= tree.target
# create calculated data matrix
entries = tree.GetEntries();
hit_calc= np.zeros((entries, 7));
for i in range(entries):
  # calculated quantities
  tree.GetEntry(i);
  hit_calc[i][0] = tree.g.GetVelocity(0);
  hit_calc[i][1] = tree.g.GetVelocity(1);
  hit_calc[i][2] = tree.g.HitSeparation(0,1);
  hit_calc[i][3] = tree.g.HitVdiff(0,1);
  hit_calc[i][4] = tree.g.HitOpeningAngle(0,1);
  hit_calc[i][5] = tree.g.HitNSI(0,1);
  hit_calc[i][6] = tree.g.HitScatteringAngle(0,1);
# copy data from the tree
entries1 = tree1.GetEntries();
hit_data1 = np.zeros((entries1, 10));
label1 = np.empty(entries1);
label1.fill(9);
for o in range(entries1):
  # get the tree index for the next entry on our list
  # and load this entry from the TTree
  tree1.GetEntry(o)
  # transfer the data for this entry to our numpy array
  # x0
  hit_data1[o][0] = tree1.g.x[0]
  # y0
  hit_data1[o][1] = tree1.g.y[0]
  # z0
  hit_data1[o][2] = tree1.g.z[0]
  # t0
  hit_data1[o][3] = tree1.g.t[0]
  # q0
  hit_data1[o][4] = tree1.g.q[0]
  # x1
  hit_data1[o][5] = tree1.g.x[1]
  # y1
  hit_data1[o][6] = tree1.g.y[1]
  # z1
  hit_data1[o][7] = tree1.g.z[1]
  # t1
  hit_data1[o][8] = tree1.g.t[1]
  # q1
  hit_data1[o][9] = tree1.g.q[1]
  # label
  label1[o]= tree1.signal
# create calculated data matrix
entries1 = tree1.GetEntries();
hit_calc1= np.zeros((entries1, 7));
for o in range(entries1):
  # calculated quantities
  tree1.GetEntry(o);
  hit_calc1[o][0] = tree1.g.GetVelocity(0);
  hit_calc1[o][1] = tree1.g.GetVelocity(1);
  hit_calc1[o][2] = tree1.g.HitSeparation(0,1);
  hit_calc1[o][3] = tree1.g.HitVdiff(0,1);
  hit_calc1[o][4] = tree1.g.HitOpeningAngle(0,1);
  hit_calc1[o][5] = tree1.g.HitNSI(0,1);
  hit_calc1[o][6] = tree1.g.HitScatteringAngle(0,1);
# Concatenate measured and calculated quantities
input_data_min_max1 = np.concatenate((hit_data1, hit_calc1), axis=1)
X1 = input_data_min_max1
y1 = label1
# Scale the features using Standered
scaler = StandardScaler()
Dataset_featured1 = scaler.fit_transform(X1)
# Concatenate measured and calculated quantities
input_data_min_max = np.concatenate((hit_data, hit_calc), axis=1)
X = input_data_min_max
y = label

# Scale the features using Standered
Dataset_featured = scaler.fit_transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(Dataset_featured,y,test_size=0.01, random_state=1)
clf = svm.SVC(kernel='linear', C=1, gamma='auto')
print(len(X_train1))
print(len(y_train1))
number = []
timearray = []
accuracy_list = []
bas_list = []
F1_list = []
Mc_list = []

for i in range(100, 20000, 500):
    start = time.time()
    selection_X1 = X_train1[0:i, :]
    selection_Y1 = y_train1[0:i]

    clf.fit(selection_X1, selection_Y1)
    y_pred = clf.predict(Dataset_featured1)
    end = time.time()
    length = end - start
    
    print(set(y1))
    print(set(y_pred))
    y1_series = pd.Series(y1)
    y_pred_series = pd.Series(y_pred)
    print(y1_series.unique())
    print(y_pred_series.unique())


    accuracy = accuracy_score(y1, y_pred)
    Mc = matthews_corrcoef(y1, y_pred)
    F1 = f1_score(y1, y_pred, average='micro')
    bas = balanced_accuracy_score(y1, y_pred)


    timearray.append(length)
    number.append(i)
    accuracy_list.append(accuracy)
    bas_list.append(bas)
    F1_list.append(F1)
    Mc_list.append(Mc)

    print("Accuracy:", accuracy)
    print("It took", length, "seconds!", i)

print(number)
print(timearray)
print("Accuracy:", accuracy_list)
print("F1 Accuracy:", F1_list)
print("bas Accuracy:", bas_list)
print("Mc Accuracy:", Mc_list)

plt.plot(number, accuracy_list, label="accuracy")
plt.plot(number, F1_list, label="F1")
plt.plot(number, bas_list, label="bas")
plt.plot(number, Mc_list, label="Mc")

plt.xlabel('number of data points')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend()
        

import time
number =[]
timearray=[]

for i in np.arange(100,40000, 100):

    # Calculate the start time
    start = time.time()

    # Code here
    selection_X1=X_train1[0:i,:]
    selection_Y1=y_train1[0:i]
    clf.fit(selection_X1,selection_Y1)
    # Calculate the end time and time taken
    end = time.time()
    length = end - start
    timearray.append(length)
    number.append(i)   
    # Show the results : this can be altered however you like
    print("It took", length, "seconds!",i)
print(number)  
print(timearray)
plt.plot(number, timearray)


# naming the x axis
plt.xlabel('number of data points')
# naming the y axis
plt.ylabel('time')

# giving a title to my graph
plt.title('number of data points affect on time')
