# Import dependencies 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10,8
sns.set_theme(style='whitegrid', palette='muted', rc={'figure.figsize': (15,10)})
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasClassifier  # using scikeras now
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from numpy.random import seed

# Load data as Pandas dataframe 
train = pd.read_csv('./train_clean.csv', ) 
test = pd.read_csv('./test_clean.csv') 
df = pd.concat([train, test], axis=0, sort=True) 
df.head()

# convert to cateogry dtype
df['Sex'] = df['Sex'].astype('category')
# convert to category codes
df['Sex'] = df['Sex'].cat.codes

# subset all categorical variables which need to be encoded
categorical = ['Embarked', 'Title']
for var in categorical:
    df = pd.concat([df, 
                    pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]
# drop the variables we won't be using
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
df.head()

# Scale continuous variable
continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']
scaler = StandardScaler()
for var in continuous:
    df[var] = df[var].astype('float64')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))
 
# Split the train and test data again    
X_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
y_train = df[pd.notnull(df['Survived'])]['Survived']
X_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)

# Define the hyperparameters and define the architecture of the ANN model.
lyrs=[8] # number of layers
act='linear'# activation function 
opt='Adam' # optimizer
dr=0.0
# set random seed for reproducibility
tf.random.set_seed(42)
seed(42)
model = Sequential()
# create first hidden layer
model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
# create additional hidden layers
for i in range(1,len(lyrs)):
    model.add(Dense(lyrs[i], activation=act))
# add dropout, default is none
model.add(Dropout(dr))
# create output layer
model.add(Dense(1, activation='sigmoid'))  # output layer
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#model = create_model()
print(model.summary())

# train model on full train set, with 80/20 CV split
training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_acc = np.mean(training.history['val_accuracy']) # val_acc
print("n%s: %.2f%%" % ('val_acc', val_acc*100))
# summarize history for accuracy
# plt.plot(training.history['acc'])
plt.plot(training.history['val_accuracy']) # val_acc
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# calculate predictions
test['Survived'] = model.predict(X_test)
test['Survived'] = test['Survived'].apply(lambda x: round(x,0)).astype('int')
solution = test[['PassengerId', 'Survived']]