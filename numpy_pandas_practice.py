"""
Contains numpy and pandas practice/refresher code snippets

"""

#%%
import numpy as np
import pandas as pd

help(np)

# practice dictionary

d = {k:x for (k,v) in {2: [1,2], 3: [2,3]}.items() for x in v}
print(d)

# generator practice
def generate_a_number(n):
    for i in range(n):
        yield i
        
mynumlist = generate_a_number(3)
for x in mynumlist:
    print(x)
    
    
#%% Pandas servies

df1 = pd.Series([1,2,3,4])
df2 = pd.Series([0, 1,2,3])

print(df1)
print(df2)

print(df1[~df1.isin(df2)]) # ~ is the not symbol, gets everything not in df2

#union 
print(pd.Series(np.union1d(df1, df2)))

print("inserection")
print(pd.Series(np.intersect1d(df1, df2)))

#delete an index
df1.drop(0)
print(df1)

#%% 
data_info = {'first' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),    
       'second' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}    
  
df = pd.DataFrame(data_info)    
df['third']=pd.Series([10,20,30],index=['a','b','c'])    
print (df)    

df["first"] = df["first"].fillna((df["first"].mean()))
print(df["first"])

#%%

l = np.random.randint(10, size=(2,3))
print(l)

#reverse
l = l[::-1]

print(l)

#min value
print(l.min())

print("Delete INsert")
print(l)
a = np.delete(l, 1, axis=1)
print(a)

#add the columnd back
a = np.insert(a, 1, np.zeros(l.shape[0]), axis=1)
print(a)

# Q: why are numpy arrays advantageous to Python lists?
# A: lists have limitations when it comes to the computation of vectorized operations which deals with element-wise multiplicaiton and addition. Lists also require information on the type of every element and results in overhead. Furthermore, NumPy arrays scale much better than Python arrays as size grows.

#%%

print("Sort array")
a = np.random.randint(5, size=(3,3))
print(a)
sorted_indices = np.argsort(a[:, 1])
sorted_arr = a[sorted_indices]
print(sorted_arr)  #sort by the second column