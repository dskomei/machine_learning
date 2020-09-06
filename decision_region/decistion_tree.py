from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.grid_search import GridSearchCV

base_dir_path = Path('./')
output_dir_path = base_dir_path.joinpath('data')
learning_data_dir_path = base_dir_path.joinpath('learning_data')

data = pd.read_csv(learning_data_dir_path.joinpath('target_user_data.csv'))
x_data = data.drop(columns=['flag_persistence', 'total_login_days', 'n_dpu'])
y_data = data['flag_persistence']

columns = x_data.columns

## 入力データの正規化
x_data_std = StandardScaler().fit_transform(x_data)
x_data_std = pd.DataFrame(x_data_std, columns=columns)

x_train, x_test, y_train, y_test = train_test_split(x_data_std,
                                                    y_data,
                                                    test_size=0.2)

model_name = 'lr'
params = [{'C': [10 ** float(i) for i in np.arange(-4, 4)]}]

clf = GridSearchCV(LogisticRegression(),
                   params,
                   cv=10,
                   scoring='accuracy')

clf.fit(x_train, y_train)

result = pd.DataFrame()
for params, mean_score, scores in clf.grid_scores_:
    tmp = pd.DataFrame([str(params), mean_score, scores.std()]).T
    tmp.columns = ['params', 'score_mean', 'score_std']
    result = pd.concat([result, tmp], axis=0)
    print('End : {}, score : {}'.format(params, mean_score))

result.to_csv(output_dir_path.joinpath(model_name + '_param_scores.tsv'), index=False, sep='\t')





