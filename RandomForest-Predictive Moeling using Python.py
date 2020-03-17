# Load Dataset - Data Understanding
import pandas as pd

df = pd.read_excel("banl.xlsx")

# Data Transformation - Data Preparation
df['target'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Descriptive Stats -  Data Understanding
## Check for missing values in each column 
df.isnull().mean().sort_values(ascending=False)*100
## Correlation between variables 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
corr = df.corr()
sns.heatmap(corr,
			xticklabels = corr.columns,
			yticklabels = corr.columns)
			
# Variable Selection -  Data Preparation

# Model - Modeling
from sklearn.cross_validation import train_test_split

train,test = trian_test_split(df1, test_size = 0.4)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

features_train = train[list(vif['features'])}
label_train = train['target']
features_test = test[list(vif['features'])}
label_test = test['target']

# Applying different algorithms to train dataset and evaluate the performance on the test data [RF, Lr, NB, NN, Gradient Boosting ]
# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RanfomForestClassifier()

clf.fit(features_train, label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train, label_train)
accuracy_test = accuracy_score(pred_test, label_test)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train),
								clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test),
                                clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr, tpr)


# Hyper parameter Tuning - Modeling - To improve the performance
from sklearn.model_selection import RandomizedSearchCV
form sklearn.ensemble import RandomForestClassifier

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(3, 10, num=1)]
max_depth.append(None)
min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(features_train, label_train)

# Final Model and Model Performance - Evaluation
## Crosstab
pd.crosstab(label_train, pd.series(pred_train), rownames = ['ACTUAL'], colnames = ['PRED'])
## ROC/AUC 
from bokeh.charts import Histogram
from ipywidgets import interact
from bokeh.plotting import figure
from bokeh.io import push_notebook, show, output_notebook
output_notebook()

from sklearn import metrics
preds = clf.predict_proba(features_train)[:,1]

fpr, tpr, _ = metrics.roc_curve(np.array(label_train), preds)
auc = metrics.auc(fpr, tpr)



