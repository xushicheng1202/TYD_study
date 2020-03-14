import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np

titanic = pandas.read_csv('DATA/泰坦尼克船员获救/titanic_train.csv')
print(titanic.describe())

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
print(titanic.describe())

print(titanic['Sex'].unique())
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

print(titanic['Embarked'].unique())
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

# 首先使用线性回归算法来进行分类
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
alg = LinearRegression()
kf = KFold(n_splits=3, shuffle=False)
predictions = []
for train, test in kf.split(titanic):
    train_predictors = titanic[predictors].iloc[train, :]
    train_target = titanic['Survived'].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)
# 查看线性回归准确率
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions == titanic['Survived']) / len(predictions)
print(accuracy)
