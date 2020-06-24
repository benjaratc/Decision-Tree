#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.โหลด csv เข้าไปใน Python Pandas

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/german_credit_data 2.csv')
df.drop('Unnamed: 0', axis = 1, inplace = True)
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว

# In[3]:


df.head(10)


# In[4]:


df.sample(10)


# In[5]:


df.tail(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[6]:


df.isnull().sum()


# In[7]:


df['Saving accounts'].value_counts()


# In[8]:


df['Saving accounts'].fillna(value = 'little', inplace = True)


# In[9]:


df['Saving accounts'].value_counts()


# In[10]:


df['Checking account'].value_counts()


# In[11]:


df['Checking account'].fillna(value = 'little', inplace = True)


# In[12]:


df['Checking account'].value_counts()


# In[13]:


df.isnull().sum()


# In[14]:


df


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[15]:


df.info()


# In[16]:


df.describe()


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้นของ features ที่สนใจ

# In[17]:


sns.pairplot(df)


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[18]:


sns.distplot(df['Duration'])


# In[19]:


sns.distplot(df['Credit amount'])


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[20]:


sns.heatmap(df.corr(), annot = df.corr())


# In[21]:


df['Risk'].replace('good',1,inplace = True)


# In[22]:


df['Risk'].replace('bad',0,inplace = True)


# In[23]:


df


# In[24]:


df1 = pd.get_dummies(df, drop_first = True)
df1


# In[25]:


plt.figure(figsize = (15,8))
sns.heatmap(df1.corr(), annot = df1.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[26]:


fig = plt.figure(figsize = (10,8))
sns.scatterplot(data = df1, x = 'Credit amount', y ='Duration')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[27]:


fig = plt.figure(figsize = (10,8))
sns.scatterplot(data = df1, x = 'Housing_rent', y ='Housing_own')


# 10. สร้าง histogram ของ feature ที่สนใจ

# In[28]:


plt.hist(df['Credit amount'])


# In[29]:


plt.hist(df['Duration'])


# 11. สร้าง box plot ของ features ที่สนใจ

# In[30]:


fig = plt.figure(figsize = (10,8))
sns.boxplot(data = df, x = 'Risk', y = 'Credit amount', orient = 'v')


# In[31]:


fig = plt.figure(figsize = (10,8))
sns.boxplot(data = df, x = 'Risk', y = 'Age', orient = 'v')


# 13. ทำ Data Visualization อื่นๆ (แล้วแต่เลือก)

# In[32]:


sns.countplot(data = df, x = 'Risk')


# In[33]:


sns.countplot(data = df, x = 'Job')


# 14. พิจารณาว่าควรทำ Normalization หรือ Standardization หรือไม่ควรทั้งสองอย่าง พร้อมให้เหตุผล 

# ควรทำ Normalization เพราะ x ไม่เป็น normal distribution

# # Default

# In[34]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score,accuracy_score


# In[35]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[36]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)


# In[37]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[38]:


predicted = dtree.predict(X_test)
predicted 


# In[39]:


confusion_matrix(y_test,predicted)


# In[40]:


print('accuracy score',accuracy_score(y_test,predicted))
print('precision score',precision_score(y_test,predicted))
print('recall_score',recall_score(y_test,predicted))
print('f1 score',f1_score(y_test,predicted))


# # Normalization

# In[41]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[42]:


min_max_scaler = MinMaxScaler()


# In[43]:


X_minmax = min_max_scaler.fit_transform(X)
X_minmax


# In[44]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y, test_size = 0.2, random_state = 100)


# In[45]:


dtree2 = DecisionTreeClassifier()
dtree2.fit(X_train,y_train)


# In[46]:


predicted2 = dtree2.predict(X_test)
predicted2 


# In[47]:


confusion_matrix(y_test,predicted2)


# In[48]:


print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2))
print('recall_score',recall_score(y_test,predicted2))
print('f1 score',f1_score(y_test,predicted2))


# # Standardization

# In[49]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[50]:


sc_X = StandardScaler()
X1 = sc_X.fit_transform(X)


# In[51]:


X_train,X_test,y_train,y_test = train_test_split(X1,y, test_size = 0.2, random_state = 100)


# In[52]:


dtree3 = DecisionTreeClassifier()
dtree3.fit(X_train,y_train)


# In[53]:


predicted3 = dtree3.predict(X_test)
predicted3 


# In[54]:


confusion_matrix(y_test,predicted3)


# In[55]:


print('accuracy score',accuracy_score(y_test,predicted3))
print('precision score',precision_score(y_test,predicted3))
print('recall_score',recall_score(y_test,predicted3))
print('f1 score',f1_score(y_test,predicted3))


# 15. เลือกช้อยที่ดีที่สุดจากข้อ 14 (หรือจะทำทุกอันแล้วนำมาเปรียบเทียบก็ได้)

# ผลของ Normalization ดีกว่า Standardization 

# 16. วัดผลโมเดล โดยใช้ confusion matrix และ ประเมินผลด้วยคะแนน Accuracy, 
# F1 score, Recall, Precision

# In[56]:


#Standardization
print('accuracy score',accuracy_score(y_test,predicted3))
print('precision score',precision_score(y_test,predicted3))
print('recall_score',recall_score(y_test,predicted3))
print('f1 score',f1_score(y_test,predicted3))


# In[57]:


#Normalization
print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2))
print('recall_score',recall_score(y_test,predicted2))
print('f1 score',f1_score(y_test,predicted2))


# In[58]:


#Default
print('accuracy score',accuracy_score(y_test,predicted))
print('precision score',precision_score(y_test,predicted))
print('recall_score',recall_score(y_test,predicted))
print('f1 score',f1_score(y_test,predicted))


# 17. หาค่า parameter combination ที่ดีที่สุด สำหรับ Dataset นี้ โดยใช้ GridSearch (Hyperparameter Tuning)

# In[59]:


from sklearn.model_selection import GridSearchCV


# In[60]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16]}


# In[61]:


grid_search = GridSearchCV(DecisionTreeClassifier(), param_combination,verbose = 3)


# In[62]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[63]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)


# In[64]:


grid_search.fit(X_train,y_train)


# In[65]:


grid_search.best_params_


# In[66]:


grid_search.best_estimator_


# In[67]:


grid_predicted = grid_search.predict(X_test)
grid_predicted 


# In[68]:


print('accuracy score',accuracy_score(y_test,grid_predicted))
print('precision score',precision_score(y_test,grid_predicted))
print('recall_score',recall_score(y_test,grid_predicted))
print('f1 score',f1_score(y_test,grid_predicted))


# 18. เลือกเฉพาะ features ที่สนใจมาเทรนโมเดล และวัดผลเปรียบเทียบกับแบบ all-features

# In[69]:


X = df1[['Credit amount','Duration','Checking account_moderate']]
y = df1['Risk']


# In[70]:


min_max_scaler2 = MinMaxScaler()


# In[71]:


X_minmax2 = min_max_scaler2.fit_transform(X)
X_minmax2


# In[72]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax2,y, test_size = 0.2, random_state = 100)


# In[73]:


dtree4 = DecisionTreeClassifier()
dtree4.fit(X_train,y_train)


# In[74]:


predicted4 = dtree4.predict(X_test)
predicted4 


# In[75]:


confusion_matrix(y_test,predicted4)


# In[76]:


print('accuracy score',accuracy_score(y_test,predicted4))
print('precision score',precision_score(y_test,predicted4))
print('recall_score',recall_score(y_test,predicted4))
print('f1 score',f1_score(y_test,predicted4))


# 19. ทำ Visualization ของค่า F1 Score ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[77]:


data = {'Default' : f1_score(y_test,predicted) , 'Grid Search': f1_score(y_test,grid_predicted),
        'Normalization' : f1_score(y_test,predicted2)}
data


# In[78]:


Series1 = pd.Series(data = data)
Series1


# In[79]:


df2 = pd.DataFrame(Series1)
df2


# In[80]:


sns.barplot(data = df2, x = df2.index, y = df2[0])
plt.ylabel('f1 score')


# 20. ทำ Visualization ของค่า Recall ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[81]:


data = {'Default' : recall_score(y_test,predicted) , 'Grid Search': recall_score(y_test,grid_predicted),
        'Normalization' : recall_score(y_test,predicted2)}
data


# In[82]:


Series2 = pd.Series(data = data)
Series2


# In[83]:


df3 = pd.DataFrame(Series2)
df3


# In[84]:


sns.barplot(data = df3, x = df3.index, y = df3[0])
plt.ylabel('recall score')


# 21. ทำ Visualization ของค่า Accuracy ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[85]:


data = {'Default' : accuracy_score(y_test,predicted) , 'Grid Search': accuracy_score(y_test,grid_predicted),
        'Normalization' : accuracy_score(y_test,predicted2)}
data


# In[86]:


Series3 = pd.Series(data = data)
Series3


# In[87]:


df4 = pd.DataFrame(Series3)
df4


# In[88]:


sns.barplot(data = df4, x = df4.index, y = df4[0])
plt.ylabel('accuracy score')


# 22. สามารถใช้เทคนิคใดก็ได้ตามที่สอนมา ใช้ Decision Tree Algorithm แล้วให้ผลลัพธ์ที่ดีที่สุดที่เป็นไปได้ (อาจจะรวม Grid Search กับ Normalization/Standardization ?)

# # Grid Search Standardization

# In[89]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[90]:


sc_X2 =  StandardScaler()
X2 = sc_X2.fit_transform(X)


# In[91]:


X_train,X_test,y_train,y_test = train_test_split(X2,y,test_size =0.3, random_state = 20)


# In[92]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16]}


# In[93]:


grid_search1 = GridSearchCV(DecisionTreeClassifier(), param_combination,verbose = 3)


# In[94]:


grid_search1.fit(X_train,y_train)


# In[95]:


grid_search1.best_params_


# In[96]:


grid_predicted1 = grid_search1.predict(X_test)
grid_predicted1


# In[97]:


confusion_matrix(y_test,grid_predicted1)


# In[98]:


print('accuracy score',accuracy_score(y_test,grid_predicted1))
print('precision score',precision_score(y_test,grid_predicted1))
print('recall_score',recall_score(y_test,grid_predicted1))
print('f1 score',f1_score(y_test,grid_predicted1))


# # Grid Search Normalization

# In[99]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[100]:


min_max_scaler3 = MinMaxScaler()


# In[101]:


X_minmax3 = min_max_scaler3.fit_transform(X)
X_minmax3


# In[102]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax3,y, test_size = 0.2, random_state = 100)


# In[103]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16]}


# In[104]:


grid_search2 = GridSearchCV(DecisionTreeClassifier(), param_combination,verbose = 3)


# In[105]:


grid_search2.fit(X_train,y_train)


# In[106]:


grid_search2.best_params_


# In[107]:


grid_predicted2 = grid_search2.predict(X_test)
grid_predicted2


# In[108]:


confusion_matrix(y_test,grid_predicted2)


# In[109]:


print('accuracy score',accuracy_score(y_test,grid_predicted2))
print('precision score',precision_score(y_test,grid_predicted2))
print('recall_score',recall_score(y_test,grid_predicted2))
print('f1 score',f1_score(y_test,grid_predicted2))

