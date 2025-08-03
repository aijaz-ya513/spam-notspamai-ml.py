#loading requierd libraries 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

#features:
#(number_links,contains_spam_words,email_length,num_uppercase_words)
x=np.array ([[1,0,180,66],[2,1,160,40],[6,0,130,25],[4,1,156,43],[5,0,156,33],[8,1,666,53],[8,0,60,56],[7,1,30,1188]])
#...........not spam,......#spam,.......#not spam,...#spam,......#not spam,....#spam,.......#not spam,...#spam.......

#labels:0=not spam,1=spam....
y=np.array([0, 1, 0, 1, 0, 1, 0, 1])

#traim model
model=LogisticRegression()
model.fit(x, y)

#model prediction 
new_email=np.array([[4,0,344,87]])
prediction=model.predict(new_email)
probablity=model.predict_proba(new_email)
print ("spam or not a spam: ",prediction[0])
print ("probablity: ", probability[0]  [1])

y_probablity=model.predict_proba(x)[:, 1]


#plotting
sns.set(style ='whitegrid')
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_probablity,y=y,color='gold',label="spam/not spam  detector")

#plotting the new email
plt.scatter(probablity[0] [1], prediction[0],color='black',s=150,marker='*',label="new email")
plt.axvline(0.23,linestyle='--',label='Decision threshold')
#labels and formating
plt.xlabel("predicted probablity spam")
plt.ylabel("actual predicted label(1=spam,0=not spam)")
plt.tight_layout()
plt.grid(True)
plt.show()
