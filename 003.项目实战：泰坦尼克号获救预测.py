import pandas
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 首先使用线性回归算法来进行分类
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
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
accuracy = sum(predictions == titanic['Survived']) / len(predictions)
print(accuracy)

# # 对于一个二分类问题来说，这个准确率似乎不太行，接下来用逻辑回归算法试下
# alg = LogisticRegression(random_state=1)
# scores = cross_val_score(alg, titanic[predictions], titanic['Survived'], cv=3)
# print(scores.mean())

# 使用随机森林算法：通过上面发现，似乎线性回归，逻辑回归这类算法似乎不太行，那这次再用随机森林算法来试下(一般来说随机森林算法比线性回归和逻辑回归算法的效果好一点)，注意随机森林参数的变化。
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = KFold(n_splits=3, shuffle=False)
scores = cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)
print(scores.mean())
# 调整随机森林的参数
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
kf = KFold(n_splits=3, shuffle=False)
scores = cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=kf)
print(scores.mean())
