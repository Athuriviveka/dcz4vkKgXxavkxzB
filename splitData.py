import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from lightgbm import LGBMClassifier

df = pd.read_csv('/users/BalakumaranSivarajan/Downloads/ACME-HappinessSurvey2020.csv')
print(df)
df.info()

print(len(df)-len(df.drop_duplicates())) #finding duplicates to remove, because duplicates leads to overfitting

df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)
print(df)

fig, ax = plt.subplots(figsize=(15,10))
corr = df.corr()
sns.heatmap(corr, annot=True)


y = df['Y']
X = df.drop(['Y'], axis=1)

# Split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
target = ['Y']

knn = KNeighborsClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
# lgbm = LGBMClassifier()

models = [knn, rf, dt]


def test_models(l):
    for model in models:
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        acc = accuracy_score(y_test, y_predict)
        print(model)
        print("\nThe Training Accuracy is : {} \n".format(model.score(X_train, y_train) * 100))
        print('The Test Accuracy is : {}\n\n'.format(acc))


test_models(models)