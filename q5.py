import pandas as pd
data = pd.read_csv("iris.data",header= None)
#data.columns=["char_a","char_b","char_c","char_d","Result"]
data_lst=[]
for i in range(150):
    temp_lst=[1]
    for j in range(5):
        temp_lst.append(data.iloc[i,j])

    data_lst.append(temp_lst)
#print(data_lst[149])

