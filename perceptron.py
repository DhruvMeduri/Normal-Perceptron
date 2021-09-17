
import q5
import numpy as np
import matplotlib.pyplot as plt
import random


def update_weights(wt,dt,learn,y):# updates the weights after every data point/iteration
    wt = wt + (learn*y)*dt
    return wt
def compute_error1(wt):# computes # points misclassified by wt wrt Setosa
    err = 0
    for i in range(150):
        temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
       # print(temp_vec)
        if q5.data_lst[i][5]!='Iris-setosa' and np.dot(wt,temp_vec)>=0:
            err = err + 1
        if q5.data_lst[i][5]=='Iris-setosa' and np.dot(wt,temp_vec)<0:
            err = err + 1
    return err
def compute_error2(wt):# computes # points misclassified by wt wrt Versicolor
    err = 0
    for i in range(150):
        temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])

        if q5.data_lst[i][5]!='Iris-versicolor' and np.dot(wt,temp_vec)>=0:
            err = err + 1
        if q5.data_lst[i][5]=='Iris-versicolor' and np.dot(wt,temp_vec)<0:
            err = err + 1
    return err

def compute_error3(wt):# computes # points misclassified by wt wrt Virginica
    err = 0
    for i in range(150):
        temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])

        if q5.data_lst[i][5]!='Iris-virginica' and np.dot(wt,temp_vec)>=0:
            err = err + 1
        if q5.data_lst[i][5]=='Iris-virginica' and np.dot(wt,temp_vec)<0:
            err = err + 1
    return err

def perceptron1(ar,learn):#This function is for the classification of Iris-Setosa>=0 and !Iris-setosa<0

   err_lst=[]
   weights = np.array([1,2,3,4,5])
   #res = list(np.random.randint(low = 1,high=6,size=5))# this is to randomize the weights
   #weights = np.array(res)

   for i in ar:
       temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
       if np.dot(weights,temp_vec)>=0 and q5.data_lst[i][5]!='Iris-setosa':
           weights = update_weights(weights,temp_vec,learn,-1)
       if np.dot(temp_vec,weights)<0 and q5.data_lst[i][5]=='Iris-setosa':
           weights = update_weights(weights,temp_vec,learn,1)
       err_lst.append(compute_error1(weights))
   print("Weights for Setosa: ",weights)
   return [weights,err_lst]

def perceptron2(ar,learn):#This function is for the classification of Iris-versicolor>=0 and !Iris-versicolor<0

   err_lst=[]
   weights = np.array([1,2,3,4,5])
   #res = list(np.random.randint(low = 1,high=6,size=5))# this is to randomize the weights
   #weights = np.array(res)
   #print(ar)
   for i in ar:
       temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
       if np.dot(weights,temp_vec)>=0 and q5.data_lst[i][5]!='Iris-versicolor':
           weights = update_weights(weights,temp_vec,learn,-1)
       if np.dot(temp_vec,weights)<0 and q5.data_lst[i][5]=='Iris-versicolor':
           weights = update_weights(weights,temp_vec,learn,1)
       err_lst.append(compute_error2(weights))

   return [weights,err_lst]

def perceptron3(ar,learn):#This function is for the classification of Iris-virginica>=0 and !Iris-virginica<0

   err_lst=[]
   weights = np.array([1,2,3,4,5])
   #res = list(np.random.randint(low = 1,high=6,size=5))# randomizing the weights
   #weights = np.array(res)
   for i in ar:
       temp_vec = np.array([q5.data_lst[i][0],q5.data_lst[i][1],q5.data_lst[i][2],q5.data_lst[i][3],q5.data_lst[i][4]])
       if np.dot(weights,temp_vec)>=0 and q5.data_lst[i][5]!='Iris-virginica':
           weights = update_weights(weights,temp_vec,learn,-1)
       if np.dot(temp_vec,weights)<0 and q5.data_lst[i][5]=='Iris-virginica':
           weights = update_weights(weights,temp_vec,learn,1)
       err_lst.append(compute_error3(weights))
   return [weights,err_lst]

#plotting error trajectory

ar=np.random.permutation(150)
result1 = perceptron1(ar,2)
result2 = perceptron2(ar,2)
result3 = perceptron3(ar,2)
'''
# this used to plot by changing learning rate
x = []
for i in range(1,151):
    x.append(i)
for j in range(1,11):
    result1 = perceptron1(ar,j/10)
    y1 = result1[1]
    plt.plot(x,y1)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Error Trajectory(Iris-setosa)-Learning Rate="+str(j/10))
    #plt.savefig("learning_rate/fig"+str(j)+".png")
    plt.show()
#print(result1)
'''
x=[]
for i in range(1,151):
    x.append(i)

# This is used to plot by shuffling the order and with random initial weights
#shuffling is done by repeating the runs many times, due to random.permutation
y1 = result1[1]
plt.plot(x,y1)
plt.xlabel("Iterations")
plt.ylabel("Error %")
plt.title("Error Trajectory(Iris-Setosa or not)")
#plt.savefig("random_weights/fig7.png")
plt.show()

y2 = result2[1]
plt.plot(x,y2)
plt.xlabel("Iterations")
plt.ylabel("Error %")
plt.title("Error Trajectory(Iris-Versicolor or not)")
#plt.savefig("random_weights/fig8.png")
plt.show()

y3 = result3[1]
plt.plot(x,y3)
plt.xlabel("Iterations")
plt.ylabel("Error %")
plt.title("Error Trajectory(Iris-Verginica or not)")
#plt.savefig("random_weights/fig9.png")
plt.show()

'''
#plotting decision boundary on parameter 1 and 2
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][2])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][2])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][2])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][2]) - (result1[0][1]/result1[0][2])*x
plt.plot(x,y)
plt.xlabel("Parameter1")
plt.ylabel("Parameter2")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/fig4.png")
plt.show()

#plotting decision boundary on parameter 1 and 3
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][3])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][3]) - (result1[0][1]/result1[0][3])*x
plt.plot(x,y)
plt.xlabel("Parameter1")
plt.ylabel("Parameter3")
plt.legend()
plt.savefig("Q5a/fig5.png")
plt.title("Decision Boundary")
plt.show()

#plotting decision boundary on parameter 1 and 4
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][4])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][1])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][4]) - (result1[0][1]/result1[0][4])*x
plt.plot(x,y)
plt.xlabel("Parameter1")
plt.ylabel("Parameter4")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/fig6.png")
plt.show()

#plotting decision boundary on parameter 2 and 3
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][3])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][3])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][3]) - (result1[0][2]/result1[0][3])*x
plt.plot(x,y)
plt.xlabel("Parameter2")
plt.ylabel("Parameter3")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/fig7.png")
plt.show()

#plotting decision boundary on parameter 2 and 4
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][4])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][2])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][4]) - (result1[0][2]/result1[0][4])*x
plt.plot(x,y)
plt.xlabel("Parameter2")
plt.ylabel("Parameter4")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/fig8.png")
plt.show()

#plotting decision boundary on parameter 3 and 4
x = []
y = []
#print(q5.data_lst)
for i in range(50):
  # print(q5.data_lst[i])
   x.append(q5.data_lst[i][3])
   y.append(q5.data_lst[i][4])
#print(x)
plt.scatter(x,y,color='red',label='Setosa')
x = []
y = []
for i in range(50,100):
   x.append(q5.data_lst[i][3])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='blue',label='Versicolor')
x = []
y = []
for i in range(100,150):
   x.append(q5.data_lst[i][3])
   y.append(q5.data_lst[i][4])
plt.scatter(x,y,color='green',label='Virginica')

#print(result1[0])
x = np.linspace(0,10,100)
y = -(result1[0][0]/result1[0][4]) - (result1[0][3]/result1[0][4])*x
plt.plot(x,y)
plt.xlabel("Parameter3")
plt.ylabel("Parameter4")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("Q5a/fig9.png")
plt.show()
'''
