import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import utils
weather_data = pd.read_csv('music_genre.csv')
X = weather_data.drop(['genre'], axis=1)
Y = weather_data['genre']
X.head()
lab = preprocessing.LabelEncoder()
model = DecisionTreeClassifier()
model.fit(X, Y)
predictions = model.predict([[13, 0]])
predictions
