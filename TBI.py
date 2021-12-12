# Library import

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


# Dada import
df = pd.read_csv('TBI.csv')


# Variables
y = df['outcome']

num = df.loc[:, ['age', 'Hb', 'platelet', 'INR', 'aPTT', 'glucose', 'MLS', 'SDH', 'EDH']]
cat = df.loc[:, ['GCS', 'motors', 'extracranial_injury', 'OP', 'pupil', '3rd_ventricle_or_basal_cistern', 'contusion', 'SAH', 'IVH']]


num.columns = ['Age', 'Hb', 'Platelet', 'INR', 'aPTT', 'Glucose', 'MLS', 'SDH', 'EDH']
cat.columns = ['GCS', 'Motor score of GCS', 'extracranial_injury', 'Operation', 'Pupil reflex', '3rd ventricle or Basal cistern', 'Contusion', 'SAH', 'IVH']

# Standardization
sc = StandardScaler()
sc.fit(num)
num_std = sc.transform(num)

num_df = pd.DataFrame(num_std)
num_df.columns = ['Age', 'Hb', 'Platelet', 'INR', 'aPTT', 'Glucose', 'MLS', 'SDH', 'EDH']

X_std = pd.concat([num_df, cat], axis=1)
X = pd.concat([num, cat], axis=1)


# Train, test set split
X_train_std, X_test_std, y_train, y_test = train_test_split(X_std, y, test_size=0.25, shuffle=True, random_state=1, stratify=y)
X_train, X_test, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=1, stratify=y)

# CRASH model data preprocess
df_crash = X_test.loc[:, ['Age', 'GCS', 'Pupil reflex', 'extracranial_injury', 'Contusion', '3rd ventricle or Basal cistern', 'SAH', 'MLS', 'Operation']]

# Age
df_crash['Age'] = df_crash['Age']-40
df_crash.loc[df_crash['Age'] < 0, 'Age'] = 0

# GCS
df_crash['GCS'] = 17 - df_crash['GCS']

# pupil
df_crash = pd.get_dummies(df_crash, columns = ['Pupil reflex'], drop_first=False)
df_crash.drop(['Pupil reflex_2'], axis = 1, inplace=True)
df_crash.rename(columns={'Pupil reflex_0':'Pupil reflex_2'}, inplace=True)

# '3rd ventricle or Basal cistern'
df_crash.loc[df_crash['3rd ventricle or Basal cistern'] == 1, '3rd ventricle or Basal cistern'] = 0
df_crash.loc[df_crash['3rd ventricle or Basal cistern'] == 2, '3rd ventricle or Basal cistern'] = 1

# MLS
df_crash.loc[df_crash['MLS'] > 0, 'MLS'] = 1

# OP
df_crash.loc[df_crash['Operation'] != 1, 'Operation'] = 3
df_crash.loc[df_crash['Operation'] == 1, 'Operation'] = 0
df_crash.loc[df_crash['Operation'] == 3, 'Operation'] = 1

# b0
df_crash['instant'] = 1

df_crash = df_crash.loc[:, ['instant', 'Age', 'GCS', 'Pupil reflex_1', 'Pupil reflex_2', 
                            'extracranial_injury', 'Contusion', '3rd ventricle or Basal cistern', 'SAH', 'MLS', 'Operation']]

# DataFrame to np.array
crash = df_crash.values

# b0, Age, GCS, pupil_1, pupil_2, extracranial_injury, petechial_hemorrhage, basal_cistern, SAH, MLS, Non-evacuated hematoma
beta14d = np.array([[-5.235039, .0704921, .1617606, .6834474, 1.389877, 
                     .4250326, .1491354, 1.475885, .3882767, .9435615, .766351]])
beta14d=beta14d.reshape(11,1)

z = crash.dot(beta14d)

# probability = 1/(1+np.exp(-z))
crash_proba=[]

for i in range(106):
  a=z[i]
  b=1/(1+ np.exp(-a))
  p=b[0]
  crash_proba.append(p)

# prediction
pred_crash = pd.DataFrame(crash_proba, columns=['CRASH'])
pred_crash[pred_crash['CRASH']>=0.5]=1
pred_crash[pred_crash['CRASH']<0.5]=0



# extracranial_injury
X_train_std.drop(['extracranial_injury'], axis = 1, inplace=True)
X_train.drop(['extracranial_injury'], axis = 1, inplace=True)
X_test_std.drop(['extracranial_injury'], axis = 1, inplace=True)
X_test.drop(['extracranial_injury'], axis = 1, inplace=True)



# Random forest
rf = RandomForestClassifier(criterion='gini', random_state=1, n_jobs=-1)

# Hyperparameter
params_rf = {'max_depth':[2,3,4], 'max_features':[4,5,6], 'min_samples_leaf':[2,3], 'min_samples_split':[2,3], 'n_estimators':[10, 50, 100]}

# GridSearchCV
grid_rf = GridSearchCV(rf, param_grid=params_rf, scoring='accuracy', cv=5, n_jobs=-1, refit=True, verbose=3)

# Train
grid_rf.fit(X_train, y_train_rf)
rf_scores_df = pd.DataFrame(grid_rf.cv_results_)
rf_scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score']]


# Train data performance
print('Random forest hyperparameter:{0}, Accuracy:{1:.3f}'.format(grid_rf.best_params_, grid_rf.best_score_))

# Test data performance
estimator_rf = grid_rf.best_estimator_
pred_rf = estimator_rf.predict(X_test)
print('Random forest accuracy: {0:.4f}'.format(accuracy_score(y_test_rf, pred_rf)))

# Random forest save as pickle file
joblib.dump(estimator_rf, './rf.pkl')



# Support vector machine
svm_linear = SVC(kernel='linear', random_state=1, probability=True)

# Hyperparameter
params_svm = {'C':[0.1, 1.0, 10.0], 'gamma':[0.1, 1.0, 10.0]}

# GridSearchCV
grid_svm_linear = GridSearchCV(svm_linear, param_grid=params_svm, scoring='accuracy', cv=5, n_jobs=-1, refit=True, verbose=3)

# Train
grid_svm_linear.fit(X_train_std, y_train)

# Train data performance
print('Support vector machine hyperparameter:{0}, Accuracy:{1:.3f}'.format(grid_svm_linear.best_params_, grid_svm_linear.best_score_))

# Test data performance
estimator_svm_linear = grid_svm_linear.best_estimator_
pred_svm_linear = estimator_svm_linear.predict(X_test_std)
print('Support vector machine accuracy: {0:.4f}'.format(accuracy_score(y_test, pred_svm_linear)))

# Support vector machine save as pickle file
joblib.dump(estimator_svm_linear, './svm.pkl')



# Logistic regression
lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', random_state=1)

# Hyperparameter
lr_parameters = {'C':[0.01, 0.1, 1, 10, 100]}

# GridSearchCV
grid_lr_l2 = GridSearchCV(lr_l2, param_grid=lr_parameters, scoring='accuracy', cv=5, refit=True, n_jobs=-1)

# Train
grid_lr_l2.fit(X_train_std, y_train)

# Train data performance
print('Logistic regression hyperparameter:{0}, Accuracy:{1:.3f}'.format(grid_lr_l2.best_params_, grid_lr_l2.best_score_))

# Test data performance
estimator_lr_l2 = grid_lr_l2.best_estimator_
pred_l2 = estimator_lr_l2.predict(X_test_std)
print('Logistic regression accuracy: {0:.4f}'.format(accuracy_score(y_test, pred_l2)))

# Logistic regression save as pickle file
joblib.dump(estimator_lr_l2, './lr.pkl')



#Early Death probability
estimator_rf_proba = estimator_rf.predict_proba(X_test)[:, 1]
estimator_svm_linear_proba = estimator_svm_linear.predict_proba(X_test_std)[:, 1]
estimator_lr_l2_proba = estimator_lr_l2.predict_proba(X_test_std)[:, 1]
estimator_crash_proba = np.array(crash_proba)

# np.arrapy -> DataFrame
df_proba_rf=pd.DataFrame(estimator_rf_proba, columns=["RF"])
df_proba_svm_linear=pd.DataFrame(estimator_svm_linear_proba, columns=["SVM_linear"])
df_proba_lr_l2=pd.DataFrame(estimator_lr_l2_proba, columns=["LR_L2"])
df_crash_proba = pd.DataFrame(crash_proba, columns=['CRASH'])

# Outcome
df_y_test=y_test.to_frame(name='Outcome')
df_y_test.reset_index(drop=True, inplace=True)

# concatenate
df_proba=pd.concat([df_y_test, 
                    df_proba_lr_l2, 
                    df_proba_rf,
                    df_proba_svm_linear,  
                    df_crash_proba], axis=1)

# DataFrame save as csv file
df_proba.to_csv('df_proba.csv', index=False)


# Performance 
# Accuracy
print('Random forest accuracy: {0:.4f}'.format(accuracy_score(y_test, pred_rf)))
print('Support vector machine accuracy: {0:.4f}'.format(accuracy_score(y_test, pred_svm_linear)))
print('Logistic regression accuracy: {0:.4f}'.format(accuracy_score(y_test, pred_l2)))
print('CRASH model accuracy: {0:.4f}'.format(accuracy_score(y_test, pred_crash)))


# Sensitivity
print('Random forest sensitivity: {0:.4f}'.format(recall_score(y_test, pred_rf)))
print('Support vector machine sensitivity: {0:.4f}'.format(recall_score(y_test, pred_svm_linear)))
print('Logistic regression sensitivity: {0:.4f}'.format(recall_score(y_test, pred_l2)))
print('CRASH model sensitivity: {0:.4f}'.format(recall_score(y_test, pred_crash)))


# Specificity
conf_rf = confusion_matrix(y_test, pred_rf)
spec_rf = conf_rf[0,0]/(conf_rf[0,0]+conf_rf[0,1])

conf_svm_linear = confusion_matrix(y_test, pred_svm_linear)
spec_svm_linear = conf_svm_linear[0,0]/(conf_svm_linear[0,0]+conf_svm_linear[0,1])

conf_l2 = confusion_matrix(y_test, pred_l2)
spec_l2 = conf_l2[0,0]/(conf_l2[0,0]+conf_l2[0,1])

conf_crash = confusion_matrix(y_test, pred_crash)
spec_crash = conf_crash[0,0]/(conf_crash[0,0]+conf_crash[0,1])

print('Random forest specificity: {0:.4f}'.format(spec_rf))
print('Support vector machine specificity: {0:.4f}'.format(spec_svm_linear))
print('Logistic regression specificity: {0:.4f}'.format(spec_l2))
print('CRASH model specificity: {0:.4f}'.format(spec_crash))



# ROC curve

def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc


plt.figure(figsize=(6.5, 6))
matplotlib.rcParams.update({'font.size': 14})
plt.grid()


# Random forest
fpr, tpr, roc_auc_rf = roc_curve_and_score(y_test, estimator_rf_proba)
plt.plot(fpr, tpr, color='dodgerblue', lw=2,
         label='Random Forest AUC={0:.3f}'.format(roc_auc_rf))

# SVM linear
fpr, tpr, roc_auc_svm_linear = roc_curve_and_score(y_test, estimator_svm_linear_proba)
plt.plot(fpr, tpr, color='crimson', lw=2,
         label='Support vector machine AUC={0:.3f}'.format(roc_auc_svm_linear))

# Logistic regression
fpr, tpr, roc_auc_l2 = roc_curve_and_score(y_test, estimator_lr_l2_proba)
plt.plot(fpr, tpr, color='gold', lw=2,
         label='Logistic regression AUC={0:.3f}'.format(roc_auc_l2))

# CRASH
fpr, tpr, roc_auc_crash = roc_curve_and_score(y_test, estimator_crash_proba)
plt.plot(fpr, tpr, color='black', lw=2,
         label='CRASH model AUC={0:.3f}'.format(roc_auc_crash))


plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':')
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim([-0.03, 1.03])
plt.ylim([-0.03, 1.03])
plt.title('ROC')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()

# AUC
print('Random Forest AUC={0:.3f}'.format(roc_auc_rf))
print('Support vector machine AUC={0:.3f}'.format(roc_auc_svm_linear))
print('Logistic regression AUC={0:.3f}'.format(roc_auc_l2))
print('CRASH model AUC={0:.3f}'.format(roc_auc_crash))


# Importance of variabless

# Random forest
importances_values_rf = estimator_rf.feature_importances_
importances = pd.Series(importances_values_rf, index=X_train.columns)
top20 = importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8, 8))
plt.title('Random Forest')
sns.barplot(x = top20, y = top20.index, palette='summer')
plt.savefig('var_rf.eps', format='eps')
plt.show()


# Support vector machine
feature_importance_svm = estimator_svm_linear.coef_.reshape(-1,)
feature_importance_svm = abs(feature_importance_svm)
importances_values_svm = feature_importance_svm
importances = pd.Series(importances_values_svm, index=X_train.columns)
top20 = importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8, 8))
plt.title('Support Vector Machine')
sns.barplot(x = top20, y = top20.index, palette='OrRd_r')
plt.savefig('var_svm.eps', format='eps')
plt.show()

# Logistic regression
feature_importance_lr = estimator_lr_l2.coef_.reshape(-1,)
feature_importance_lr = abs(feature_importance_lr)
importances_values_lr = feature_importance_lr
importances = pd.Series(importances_values_lr, index=X_train.columns)
top20 = importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8, 8))
plt.title('Logistic Regression')
sequential_colors = sns.color_palette("RdPu", 10)
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.barplot(x = top20, y = top20.index, palette='Blues_r')
plt.savefig('var_LR.eps')
plt.show()

