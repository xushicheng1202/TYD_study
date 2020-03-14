# 模块引入
import matplotlib.pyplot as plt
import pandas as pd
# 数据集通过sklearn模块内置的 房价预测数据集
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
from graphviz import Digraph
import pydotplus
from IPython.display import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.data[0])

# 实例化树模型，指定最大深度，构造决策树，传入参数分别表示 x 和 y 值
dtr = tree.DecisionTreeRegressor(max_depth=2)
dtr.fit(housing.data[:, [6, 7]], housing.target)
# 可视化处理，指定配置参数，其它几个参数就不需要改动了
dot_data = \
    tree.export_graphviz(
        dtr,  # 实例化的决策树
        out_file=None,
        feature_names=housing.feature_names[6:8],  # 要用到的特征
        filled=True,
        impurity=False,
        rounded=True
    )
# 可视化生成，颜色可以随意改动，其它不动就可以了
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor('#FFF2DD')
img = Image(graph.create_png())
graph.write_png('out.png')  # 视情况可以进行保存等操作，保存为png格式更加清晰一点

# 切分数据集
data_train, data_test, target_train, target_test = train_test_split(housing.data, housing.target, test_size=0.1,
                                                                    random_state=42)
dtr = tree.DecisionTreeRegressor(random_state=42)
dtr.fit(data_train, target_train)
print('dtr.score:', dtr.score(data_test, target_test))

# 选择参数
rfr = RandomForestRegressor(random_state=42)
rfr.fit(data_train, target_train)
print('rfr.score:', rfr.score(data_test, target_test))

tree_param_grid = {'min_samples_split': list((3, 6, 9)), 'n_estimators': list((10, 50, 100))}
grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print('grid.cv_results_:', grid.cv_results_)
print('grid.best_params_:', grid.best_params_)
print('grid.best_score_:', grid.best_score_)
