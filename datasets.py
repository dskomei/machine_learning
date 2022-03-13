import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns



#
# データ部
# データを読み込んだ後、入力・出力データに分けている
#
data = datasets.load_iris()
x_data = data.data
y_data = data.target

# データを学習用/テスト用に分割している
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.2)
#
# 学習部
# 今回は学習器として『決定木』を使っている
#
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
model.fit(x_train, y_train)

#
# 測定部
# 正答率を測定している
#
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
print('train accuracy score : {:.0f}%'.format(train_score*100))
print('test accuracy score : {:.0f}%'.format(test_score*100))



lr = Pipeline([('scl', StandardScaler()),
               ('clf', LogisticRegression(C=10))])

knn = Pipeline([('scl', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=5))])

svm = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(kernel='rbf', C=1.0))])

dc = DecisionTreeClassifier(criterion='entropy', max_depth=3)


rf = RandomForestClassifier(criterion='entropy',
                            n_estimators=10)

models = [lr, knn, svm, dc, rf]
model_names = ['logistic regression',
               'k nearest neighbor',
               'svm',
               'decision tree',
               'random forest']


for model_name, model in zip(model_names, models):

    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print('Model : {:20}, train accuracy : {:3.0f}, test accuracy : {:3.0f}'.format(model_name,
                                                                                    train_score*100,
                                                                                    test_score*100))
## 複数回の試行によりモデルを評価する
score_datas = pd.DataFrame()
for loop in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.2)

    for model_name, model in zip(model_names, models):
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        print('loop {}, Model : {:20}, train accuracy : {:3.0f}, test accuracy : {:3.0f}'.format(loop,
                                                                                                 model_name,
                                                                                                 train_score * 100,
                                                                                                 test_score * 100))

        score_datas = pd.concat([score_datas,
                                 pd.DataFrame({'number':[loop],
                                               'model_name':[model_name],
                                               'score_name':['train'],
                                               'score':[train_score]}),
                                 pd.DataFrame({'number': [loop],
                                               'model_name': [model_name],
                                               'score_name': ['test'],
                                               'score': [test_score]})])
score_datas_sum = score_datas.groupby(['score_name', 'model_name']).agg({'score':[np.mean, np.std]})
print(score_datas_sum)

fig = sns.factorplot(x='model_name',
                     y='score',
                     col='score_name',
                     data=score_datas,
                     kind='swarm')
fig.set_xlabels('')
fig.set_xticklabels(rotation=40)
plt.tight_layout()
plt.savefig('./images/scores.png', dpi=300)
plt.show()

