# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#data_path = './winequalityN.csv'
#HI 
data_path = r'winequalityN.csv'
wine = pd.read_csv(data_path)


'''
Some things to consider from the paper...

Data was first standardized to a zero mean and one standard deviation
The use of regression error characteristic curve for evaluation
The use of mean absolute deviation for regression evaluation
The use of precision for evaluation
SVM with gaussian kernel using γ, ε and C
Sensitivity analysis for attribute pruning
Classification of quality 3 and 9 were impossible for the authors - why is that?
SVM relative importance plots for attributes
SVM hyperparameter tuning
“We will adopt a regression approach, which preserves the order of the preferences. For instance, if the true grade is 3, then a model that predicts 4 is better than one that predicts 7.”


'''



# %%
## Data preprocessing and cleaning

# explore whole data set
print(wine.info())
print(wine.describe())

# splitting  red/white datasets
white = wine[wine['type'] == 'white']
red = wine[wine['type'] == 'red']

# removing 'type' columns
white = white.drop(columns = ['type'])
red = red.drop(columns = ['type'])

# dropping records w/ missing values
white = white.dropna()
red = red.dropna()

# TODO: remove 3 and 9 quality wines - N
# TODO: standardized to a zero mean and one standard deviation for input attr - N



# %%
## Exploratory Analysis

# Data summary (mean, median, standard deviation, etc.)
# White wine
print("\nWhite Wine\n")
print(white.describe())
print(white.info())

# Red wine
print("\n\nRed Wine\n")
print(red.describe())
print(red.info())





# %%
## Data visualizations

# Histograms
# white wine
plt.title('Histogram of White Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.hist(white['quality'], bins=[3,4,5,6,7,8,9,10])
sns.set(style="whitegrid")
plt.show()
plt.clf()
# red wine
plt.title('Histogram of Red Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.hist(red['quality'], bins=[3,4,5,6,7,8,9])
sns.set(style="whitegrid")
plt.show()
plt.clf()
# appears that quality of both red and white wine follows roughly a normal distribution



# Correlation maps
# white wine
plt.figure(figsize=(10,6))
corr_white = sns.heatmap(white.corr(), annot=True, cmap='cubehelix_r')
corr_white.set_title('Correlation Map - White Wine')
plt.show()
plt.clf()
# red wine
plt.figure(figsize=(10,6))
corr_red = sns.heatmap(red.corr(), annot=True, cmap='cubehelix_r')
corr_red.set_title('Correlation Map - Red Wine')
plt.show()
plt.clf()
# alcohol content has highest correlation with quality for both red and white wine




# %%
### Train/Test Split and other Utilities
# Paper did a 2/3 1/3 split

# It eventually won't matter when we do k-fold cross validation
# but it can easily be adjusted for preliminary models with test_size=0.33 - Mathew
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
# labels for confusion matrices
# TODO: fix labels, remove 3 and 9 - N
white_labels = [3, 4, 5, 6, 7, 8, 9]
red_labels = [3, 4, 5, 6, 7, 8]
test_size = 0.33

# Red wine
red_x = red.drop(['quality'], axis=1)
red_y = red['quality']
red_train_x, red_test_x, red_train_y, red_test_y = train_test_split(red_x, red_y, test_size=test_size, random_state=42, shuffle=True, stratify=red_y)
red_train_y = red_train_y.to_numpy()
red_test_y = red_test_y.to_numpy()

# White wine
white_x = white.drop(['quality'], axis=1)
white_y = white['quality']
white_train_x, white_test_x, white_train_y, white_test_y = train_test_split(white_x, white_y, test_size=test_size, random_state=42, shuffle=True, stratify=white_y)
white_train_y = white_train_y.to_numpy()
white_test_y = white_test_y.to_numpy()

# Models and metrics lists for later plotting/comparison
models_white = []
accuracy_white = []
SSE_white = []
models_red = []
accuracy_red = []
SSE_red = []

# TODO: add in functions for precision, MAD, and REC - M
# TODO: 2D representation of classifier splitting data - J

# %%
def two_dimensional_representation(x_data,y_data,title="t-SNE wine"):
    tsne = TSNE(verbose=1, perplexity=50, random_state = 42)
    X_embedded_data= tsne.fit_transform(x_data)


    # sns settings
    sns.set(rc={'figure.figsize':(8,8)})

    # colors
    palette = sns.color_palette("bright", 1)

    real = white_test_y
    palette = sns.color_palette("bright", len(set(real)))
    print(real)
    # plot
    sns.scatterplot(X_white[:,0], X_white[:,1], hue = real, palette=palette)

    plt.title(title)
    # plt.savefig("plots/t-sne_wine.png")
    plt.show()

two_dimensional_representation(red_test_x,red_test_y,"Red Default")
#%%
'''
# defining SSE
def SSE(actual, pred):
    s = 0
    for i in range(len(actual)):
        s += abs(actual[i]-pred[i])**2
    return s


def ROC_AUC(y_pred, y_true, pos_group=None):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, pos_label=pos_group)
    auc_result = metrics.auc(fpr, tpr)
    return fpr, tpr, auc_result


def plotROCAUC(df, df2, classes, micro_fpr, micro_tpr, micro_roc_auc):
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate( [df2['fpr'][i] for i in classes] ))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in classes:
        mean_tpr += np.interp(all_fpr, df2['fpr'][i], df2['tpr'][i])
    # Finally average it and compute AUC
    mean_tpr /= len(classes)
    macro_fpr = all_fpr
    macro_tpr = mean_tpr
    macro_roc_auc = metrics.auc(macro_fpr, macro_tpr)
    # Plot ROC AUC curves
    lw = 2
    plt.figure(figsize=(10,6))
    plt.plot(micro_fpr, micro_tpr, label='micro-average ROC curve (area = {0:0.2f})'.format(micro_roc_auc),
         color='deeppink', linestyle=':', linewidth=4)
    plt.plot(macro_fpr, macro_tpr, label='macro-average ROC curve (area = {0:0.2f})'.format(macro_roc_auc),
         color='crimson', linestyle=':', linewidth=4)
    colors = ['aqua', 'darkorange', 'cyan', 'lightcoral', 'olive', 'fuchsia', 'indigo']
    for i, color in zip(classes, colors[:len(classes)]):
        plt.plot(df2['fpr'][i], df2['tpr'][i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'.format(i, df['roc auc'][i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def PR_AUC(y_pred, y_true, pos_group=None):
    avg_precision = metrics.average_precision_score(y_true=y_true, y_score=y_pred, pos_label=pos_group)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_pred, pos_label=pos_group)
    return precision, recall, avg_precision


def plotPRAUC(df, df2, classes, y_test, y_score, micro_precision, micro_recall, micro_avg_precision):
    # honeslty I don't know if this is portable or correct...meh
    # First aggregate all precision
    all_precision = np.unique(np.concatenate( [df2['precision'][i] for i in classes] ))
    # Then interpolate all pr curves at this points
    mean_recall = np.zeros_like(all_precision)
    for i in classes:
        mean_recall += np.interp(all_precision, df2['precision'][i], df2['recall'][i])
    # Finally average it and compute AUC
    mean_recall /= len(classes)
    macro_precision = all_precision
    macro_recall = mean_recall    
    macro_avg_precision = metrics.average_precision_score(y_true=y_test, y_score=y_score, pos_label=1)
    # macro_avg_precision_auc = metrics.auc(macro_precision, macro_recall)
    # print(macro_avg_precision_auc, macro_avg_precision)
    # average precision, also know as precision-recall area under the curve
    # Plot PR AUC curves
    lw = 2
    plt.figure(figsize=(10,6))
    plt.plot(micro_precision, micro_recall, label='micro-average PR curve (area = {0:0.2f})'.format(micro_avg_precision),
         color='deeppink', linestyle=':', linewidth=4)
    plt.plot(macro_precision, macro_recall, label='macro-average PR curve (area = {0:0.2f})'.format(macro_avg_precision),
         color='crimson', linestyle=':', linewidth=4)
    colors = ['aqua', 'darkorange', 'cyan', 'lightcoral', 'olive', 'fuchsia', 'indigo']
    for i, color in zip(classes, colors[:len(classes)]):
        plt.plot(df2['precision'][i], df2['recall'][i], color=color, lw=lw,
                label='PR curve of class {0} (area = {1:0.2f})'.format(i, df['pr avg prec'][i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall to multi-class')
    plt.legend(loc="lower right")
    plt.show()


# Learn to predict each class against the other
def oneVsRestAnalysis(model, X_train, y_train, X_test, y_test, classes):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y_train = label_binarize(y_train, classes=classes)
    y_test = label_binarize(y_test, classes=classes)
    # create classifier for multi-label classification
    clf = OneVsRestClassifier(model)
    y_score = clf.fit(X_train, y_train).decision_function(X_test)

    # Compute performance metrics
    curve_dict = { 'quality': [], 'fpr': [], 'tpr': [], 'precision': [], 'recall': [] }
    df_dict = { 'quality': [], 'roc auc': [], 'pr avg prec': [] } 
    for i,x in enumerate(classes):
        df_dict['quality'].append(x)
        curve_dict['quality'].append(x)
        # curves can't handle mutli-label, so only give column that correpsonds to binarized label with [:, i]
        fpr, tpr, roc_auc = ROC_AUC(y_true=y_test[:, i], y_pred=y_score[:, i], pos_group=1)
        curve_dict['fpr'].append(fpr)
        curve_dict['tpr'].append(tpr)
        df_dict['roc auc'].append(roc_auc)

        prec, recall, pr_avg_prec = PR_AUC(y_true=y_test[:, i], y_pred=y_score[:, i], pos_group=1)
        curve_dict['precision'].append(prec)
        curve_dict['recall'].append(recall)
        df_dict['pr avg prec'].append(pr_avg_prec)

    # Compute micro-averages 
    micro_fpr, micro_tpr, micro_roc_auc = ROC_AUC(y_true=y_test.ravel(), y_pred=y_score.ravel(), pos_group=1)
    micro_precision, micro_recall, micro_avg_precision = PR_AUC(y_true=y_test.ravel(), y_pred=y_score.ravel(), pos_group=1)

    # Analyze roc_auc and pr_auc for the classes
    df = pd.DataFrame.from_dict(df_dict)
    df.set_index('quality', inplace=True)

    df2 = pd.DataFrame.from_dict(curve_dict)
    df2.set_index('quality', inplace=True)

    plotROCAUC(df, df2, classes, micro_fpr, micro_tpr, micro_roc_auc)

    plotPRAUC(df, df2, classes, y_test, y_score, micro_precision, micro_recall, micro_avg_precision)
'''


# %%
# trivial classifier 
# predicts the mode (6 for white records, 5 for red) for all records
# this will serve as baseline for performance
# TODO: make function - N
from scipy import stats

white_y_pred = np.full(white_test_y.shape, stats.mode(white_train_y)[0]) # need to change these to the modes of the training sets
red_y_pred = np.full(red_test_y.shape, stats.mode(red_train_y)[0])

white_trivial_acc = accuracy_score(white_test_y, white_y_pred)
red_trivial_acc = accuracy_score(red_test_y, red_y_pred)
white_trivial_SSE = SSE(white_test_y, white_y_pred)
red_trivial_SSE = SSE(red_test_y, red_y_pred)

print (f" Accuracy for trivial classifier on white dataset is {white_trivial_acc}")
print (f" SSE for trivial classifier on white dataset is {white_trivial_SSE}")
data = confusion_matrix(white_test_y, white_y_pred, labels = white_labels)
white_trivial_df_cm = pd.DataFrame(data, columns = white_labels, index = white_labels)
white_trivial_df_cm.index.name = 'Actual'
white_trivial_df_cm.columns.name = 'Predicted'
white_trivial_cm = sns.heatmap(white_trivial_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
white_trivial_cm.tick_params(left = False, bottom = False)
white_trivial_cm.set_title('Trivial Classifier - White Wine')
white_trivial_cm
plt.show()
plt.clf()

print (f" Accuracy for trivial classifier on red dataset is {red_trivial_acc}")
print (f" SSE for trivial classifier on red dataset is {red_trivial_SSE}")
data = confusion_matrix(red_test_y, red_y_pred, labels = red_labels)
red_trivial_df_cm = pd.DataFrame(data, columns = red_labels, index = red_labels)
red_trivial_df_cm.index.name = 'Actual'
red_trivial_df_cm.columns.name = 'Predicted'
red_trivial_cm = sns.heatmap(red_trivial_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
red_trivial_cm.tick_params(left = False, bottom = False)
red_trivial_cm.set_title('Trivial Classifier - Red Wine')
red_trivial_cm
plt.show()
plt.clf()




# %%
## Replicate Author's SVM
# Support vector machines
# use LIBSVM  ?
# TODO: make function and try to replicate - M
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
parameters = {'C':range(1,20), 'gamma':['scale', 'auto']}
svm = svm.SVR(kernel='rbf', cache_size=400)
clf = RandomizedSearchCV(svm, parameters, n_jobs=1, n_iter=10, verbose=True, random_state=42, cv=3)
clf.fit(white_train_x, white_train_y)
print(clf.best_params_)



# %%
white_y_pred = rbf.predict(white_test_x)
white_rbf_acc = accuracy_score(white_test_y, white_y_pred)
white_rbf_SSE = SSE(white_test_y, white_y_pred)
print (f" Accuracy for rbf SVM on white dataset is {white_rbf_acc}")
print (f" SSE for rbf SVM on white dataset is {white_rbf_SSE}")

models_white.append('SVM')
accuracy_white.append(white_rbf_acc)
SSE_white.append(white_rbf_SSE)

data = confusion_matrix(white_test_y, white_y_pred, labels = white_labels)
white_rbf_df_cm = pd.DataFrame(data, columns = white_labels, index = white_labels)
white_rbf_df_cm.index.name = 'Actual'
white_rbf_df_cm.columns.name = 'Predicted'
white_rbf_cm = sns.heatmap(white_rbf_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
white_rbf_cm.tick_params(left = False, bottom = False)
white_rbf_cm.set_title('SVM Classifier - White Wine')
plt.show()
plt.clf()

#%%
# rbf.fit(red_train_x,red_train_y)

# red_y_pred = rbf.predict(red_test_x)
# red_rbf_acc = accuracy_score(red_test_y, red_y_pred)
# red_rbf_SSE = SSE(red_test_y, red_y_pred)
# print (f" Accuracy for rbf SVM on red dataset is {red_rbf_acc}")
# print (f" SSE for rbf SVM on red dataset is {red_rbf_SSE}")

# models_red.append('SVM')
# accuracy_red.append(red_rbf_acc)
# SSE_red.append(red_rbf_SSE)

# data = confusion_matrix(red_test_y, red_y_pred, labels = red_labels)
# red_rbf_df_cm = pd.DataFrame(data, columns = red_labels, index = red_labels)
# red_rbf_df_cm.index.name = 'Actual'
# red_rbf_df_cm.columns.name = 'Predicted'
# red_rbf_cm = sns.heatmap(red_rbf_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
# red_rbf_cm.tick_params(left = False, bottom = False)
# red_rbf_cm.set_title('SVM Classifier - Red Wine')
# plt.show()
# plt.clf()




'''# %%
# "Base" classifiers

from sklearn.linear_model import LogisticRegression as LR
print('LogisticRegression')
model = LR(random_state=0, n_jobs=-1)
oneVsRestAnalysis(model, white_train_x, white_train_y, white_test_x, white_test_y, white_labels)

from sklearn.linear_model import RidgeClassifier as RC
print('RidgeClassifier')
model = RC(random_state=0)
oneVsRestAnalysis(model, white_train_x, white_train_y, white_test_x, white_test_y, white_labels)
# logistic and ridge regression perform the best

'''



## Unsupervised Learning

# %%
# # Clustering
# from sklearn.cluster import KMeans
# TODO: make function - J

# from sklearn.neighbors import NearestNeighbors as NN # this is unsupervised version
# TODO: make function - M


## Supervised Learning
# %%
# Decision tree
from sklearn import tree
# TODO: make function - N
dectree = tree.DecisionTreeClassifier(max_depth = 10, random_state = 42)

white_dectree = dectree.fit(white_train_x, white_train_y)

#white_tree = tree.plot_tree(white_dectree)
white_y_pred = white_dectree.predict(white_test_x)
white_dectree_acc = accuracy_score(white_test_y, white_y_pred)
white_dectree_SSE = SSE(white_test_y, white_y_pred)

print (f" Accuracy for decision tree on white dataset is {white_dectree_acc}")
print (f" SSE for decision tree on white dataset is {white_dectree_SSE}")

models_white.append('DT')
accuracy_white.append(white_dectree_acc)
SSE_white.append(white_dectree_SSE)

data = confusion_matrix(white_test_y, white_y_pred, labels = white_labels)
white_tree_df_cm = pd.DataFrame(data, columns = white_labels, index = white_labels)
white_tree_df_cm.index.name = 'Actual'
white_tree_df_cm.columns.name = 'Predicted'
white_tree_cm = sns.heatmap(white_tree_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
white_tree_cm.tick_params(left = False, bottom = False)
white_tree_cm.set_title('Decision Tree Classifier - White Wine')
plt.show()
plt.clf()

# %%
red_dectree = dectree.fit(red_train_x, red_train_y)
#red_tree = tree.plot_tree(red_dectree)
red_y_pred = red_dectree.predict(red_test_x)
red_dectree_acc = accuracy_score(red_test_y, red_y_pred)
red_dectree_SSE = SSE(red_test_y, red_y_pred)

print (f" Accuracy for decision tree on red dataset is {red_dectree_acc}")
print (f" SSE for decision tree on red dataset is {red_dectree_SSE}")

models_red.append('DT')
accuracy_red.append(red_dectree_acc)
SSE_red.append(white_dectree_SSE)

data = confusion_matrix(red_test_y, red_y_pred, labels = red_labels)
red_tree_df_cm = pd.DataFrame(data, columns = red_labels, index = red_labels)
red_tree_df_cm.index.name = 'Actual'
red_tree_df_cm.columns.name = 'Predicted'
red_tree_cm = sns.heatmap(red_tree_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
red_tree_cm.tick_params(left = False, bottom = False)
red_tree_cm.set_title('Decision Tree Classifier - Red Wine')
plt.show()
plt.clf()






# %%
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# TODO: make function - J
gnb = GaussianNB()
white_y_pred = gnb.fit(white_train_x, white_train_y).predict(white_test_x)
white_gnb_acc = accuracy_score(white_test_y, white_y_pred)
white_gnb_SSE = SSE(white_test_y, white_y_pred)

print (f" Accuracy for Gaussian naive Bayes on white dataset is {white_gnb_acc}")
print (f" SSE for Gaussian naive Bayes on white dataset is {white_gnb_SSE}")

models_white.append('GNB')
accuracy_white.append(white_gnb_acc)
SSE_white.append(white_gnb_SSE)

data = confusion_matrix(white_test_y, white_y_pred, labels = white_labels)
white_gnb_df_cm = pd.DataFrame(data, columns = white_labels, index = white_labels)
white_gnb_df_cm.index.name = 'Actual'
white_gnb_df_cm.columns.name = 'Predicted'
white_gnb_cm = sns.heatmap(white_gnb_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
white_gnb_cm.tick_params(left = False, bottom = False)
white_gnb_cm.set_title('Gaussian Naive Bayes Classifier - White Wine')
plt.show()
plt.clf()

#%%
red_y_pred = gnb.fit(red_train_x, red_train_y).predict(red_test_x)
red_gnb_acc = accuracy_score(red_test_y, red_y_pred)
red_gnb_SSE = SSE(red_test_y, red_y_pred)

print (f" Accuracy for Gaussian naive Bayes on red dataset is {red_gnb_acc}")
print (f" SSE for Gaussian naive Bayes on red dataset is {red_dectree_SSE}")

models_red.append('GNB')
accuracy_red.append(red_gnb_acc)
SSE_red.append(red_gnb_SSE)

data = confusion_matrix(red_test_y, red_y_pred, labels = red_labels)
red_gnb_df_cm = pd.DataFrame(data, columns = red_labels, index = red_labels)
red_gnb_df_cm.index.name = 'Actual'
red_gnb_df_cm.columns.name = 'Predicted'
red_gnb_cm = sns.heatmap(red_gnb_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
red_gnb_cm.tick_params(left = False, bottom = False)
red_gnb_cm.set_title('Gaussian Naive Bayes Classifier - Red Wine')
plt.show()
plt.clf()





#%%
# K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# TODO: make function - N baby
#What N?
# maybe total classes -1 Play around a little
white_k = []
white_knn_accs = []
white_knn_SSEs = []

scaler = StandardScaler()
scaled_white_train_x = scaler.fit_transform(white_train_x)
scaled_white_test_x = scaler.fit_transform(white_test_x)

for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(scaled_white_train_x,white_train_y)
    knn_pred = knn.predict(scaled_white_test_x)
    acc = accuracy_score(white_test_y, knn_pred)
    s = SSE(white_test_y, knn_pred)
    white_k.append(i)
    white_knn_accs.append(acc)
    white_knn_SSEs.append(s)

plt.plot(white_k, white_knn_accs)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Values of k - White Wine')
plt.xticks([1,5,10,15,20])
plt.yticks([0.5, 0.55, 0.6, 0.65])
plt.show()
plt.clf()

plt.plot(white_k, white_knn_SSEs)
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('SSE for Different Values of k - White Wine')
plt.xticks([1,5,10,15,20])
plt.show()
plt.clf()
# k=1 performs best

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(scaled_white_train_x,white_train_y)
white_y_pred = knn.predict(scaled_white_test_x)
white_knn_acc = accuracy_score(white_test_y, white_y_pred)
white_knn_SSE = SSE(white_test_y, white_y_pred)
print (f" Accuracy for 1-NN on white dataset is {white_knn_acc}")
print (f" SSE for 1-NN on white dataset is {white_knn_SSE}")

models_white.append('KNN')
accuracy_white.append(white_knn_acc)
SSE_white.append(white_knn_SSE)

data = confusion_matrix(white_test_y, white_y_pred, labels = white_labels)
white_knn_df_cm = pd.DataFrame(data, columns = white_labels, index = white_labels)
white_knn_df_cm.index.name = 'Actual'
white_knn_df_cm.columns.name = 'Predicted'
white_knn_cm = sns.heatmap(white_knn_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
white_knn_cm.tick_params(left = False, bottom = False)
white_knn_cm.set_title('1-NN Classifier - White Wine')
plt.show()
plt.clf()

# %%
red_k = []
red_knn_accs = []
red_knn_SSEs = []

scaled_red_train_x = scaler.fit_transform(red_train_x)
scaled_red_test_x = scaler.fit_transform(red_test_x)

for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(scaled_red_train_x,red_train_y)
    knn_pred = knn.predict(scaled_red_test_x)
    acc = accuracy_score(red_test_y, knn_pred)
    s = SSE(red_test_y, knn_pred)
    red_k.append(i)
    red_knn_accs.append(acc)
    red_knn_SSEs.append(s)

plt.plot(red_k, red_knn_accs)
plt.title('Accuracy for Different Values of k - Red Wine')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks([1,5,10,15,20])
plt.yticks([0.5, 0.55, 0.6, 0.65])
#plt.yticks([0.52,0.53,0.54,0.55,0.56,0.57, 0.58, 0.59, 0.60,0.61,0.62,0.63])
plt.show()
plt.clf()
# k=1 and k = 4 perform best--using k=1 for consistency w/ white wine KNN

plt.plot(red_k, red_knn_SSEs)
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('SSE for Different Values of k - Red Wine')
plt.xticks([1,5,10,15,20])
plt.show()
plt.clf()

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(scaled_red_train_x,red_train_y)
red_y_pred = knn.predict(scaled_red_test_x)
red_knn_acc = accuracy_score(red_test_y, red_y_pred)
red_knn_SSE = SSE(red_test_y, red_y_pred)
print (f" Accuracy for 1-NN on red dataset is {red_knn_acc}")
print (f" SSE for 1-NN on red dataset is {red_knn_SSE}")

models_red.append('KNN')
accuracy_red.append(red_knn_acc)
SSE_red.append(red_knn_SSE)

data = confusion_matrix(red_test_y, red_y_pred, labels = red_labels)
red_knn_df_cm = pd.DataFrame(data, columns = red_labels, index = red_labels)
red_knn_df_cm.index.name = 'Actual'
red_knn_df_cm.columns.name = 'Predicted'
red_knn_cm = sns.heatmap(red_knn_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
red_knn_cm.tick_params(left = False, bottom = False)
red_knn_cm.set_title('1-NN Classifier - Red Wine')
plt.show()
plt.clf()






#%%
# Support vector machines
from sklearn import svm
# TODO: make function - J
rbf = svm.SVC(kernel = 'rbf', random_state = 42)
rbf.fit(white_train_x,white_train_y)

white_y_pred = rbf.predict(white_test_x)
white_rbf_acc = accuracy_score(white_test_y, white_y_pred)
white_rbf_SSE = SSE(white_test_y, white_y_pred)
print (f" Accuracy for rbf SVM on white dataset is {white_rbf_acc}")
print (f" SSE for rbf SVM on white dataset is {white_rbf_SSE}")

models_white.append('SVM')
accuracy_white.append(white_rbf_acc)
SSE_white.append(white_rbf_SSE)

data = confusion_matrix(white_test_y, white_y_pred, labels = white_labels)
white_rbf_df_cm = pd.DataFrame(data, columns = white_labels, index = white_labels)
white_rbf_df_cm.index.name = 'Actual'
white_rbf_df_cm.columns.name = 'Predicted'
white_rbf_cm = sns.heatmap(white_rbf_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
white_rbf_cm.tick_params(left = False, bottom = False)
white_rbf_cm.set_title('SVM Classifier - White Wine')
plt.show()
plt.clf()

#%%
rbf.fit(red_train_x,red_train_y)

red_y_pred = rbf.predict(red_test_x)
red_rbf_acc = accuracy_score(red_test_y, red_y_pred)
red_rbf_SSE = SSE(red_test_y, red_y_pred)
print (f" Accuracy for rbf SVM on red dataset is {red_rbf_acc}")
print (f" SSE for rbf SVM on red dataset is {red_rbf_SSE}")

models_red.append('SVM')
accuracy_red.append(red_rbf_acc)
SSE_red.append(red_rbf_SSE)

data = confusion_matrix(red_test_y, red_y_pred, labels = red_labels)
red_rbf_df_cm = pd.DataFrame(data, columns = red_labels, index = red_labels)
red_rbf_df_cm.index.name = 'Actual'
red_rbf_df_cm.columns.name = 'Predicted'
red_rbf_cm = sns.heatmap(red_rbf_df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
red_rbf_cm.tick_params(left = False, bottom = False)
red_rbf_cm.set_title('SVM Classifier - Red Wine')
plt.show()
plt.clf()





# %%
# # Neural network - might not have great performance
# TODO: make function - M
# #max_iter default is 200
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(random_state = 42, max_iter = 200)
# clf_pred = clf.predict(white_test_x)
# print (f" Accuracy for neural_network is {accuracy_score(white_test_y, clf_pred)}")
# print(confusion_matrix(white_test_y, clf_pred, labels = white_labels))




# ### Model evaluation/tuning
# %%
# ## feature selection
# TODO: make function, maybe use PCA? - N

# ## retrain with shuffled stratified K-fold cross validation
# %%
# TODO: clean up, make function - N
# bar graphs to compare performance of classifiers

data = pd.DataFrame(list(zip(models_white, accuracy_white)), 
               columns =['Model', 'Accuracy'])

sns.barplot(data=data, x="Model", y="Accuracy")
sns.set(style="whitegrid")
sns.despine(left=True)
plt.title('Classification Accuracy - White Wine')
for i in range(len(accuracy_white)+1):
    plt.text(x=i-1, y=0.05, s=format(accuracy_white[i-1], '1f'), 
                 color='#FFFFFF', fontsize=13, horizontalalignment='center')
plt.axhline(y = white_trivial_acc, color='k', linestyle="--")
plt.ylim(0,0.7)
plt.show()
plt.clf()
data = pd.DataFrame(list(zip(models_red, accuracy_red)), 
               columns =['Model', 'Accuracy'])

sns.barplot(data=data, x="Model", y="Accuracy")
sns.set(style="whitegrid")
sns.despine(left=True)
plt.title('Classification Accuracy - Red Wine')
for i in range(len(accuracy_red)+1):
    plt.text(x=i-1, y=0.05, s=format(accuracy_red[i-1], '1f'), 
                 color='#FFFFFF', fontsize=13, horizontalalignment='center')
plt.axhline(y = red_trivial_acc, color='k', linestyle="--")
plt.ylim(0,0.7)
plt.show()
plt.clf()

data = pd.DataFrame(list(zip(models_white, SSE_white)), 
               columns =['Model', 'SSE'])

sns.barplot(data=data, x="Model", y="SSE")
sns.set(style="whitegrid")
sns.despine(left=True)
plt.title('Classification SSE - White Wine')
for i in range(len(SSE_white)+1):
    plt.text(x=i-1, y=100, s=format(SSE_white[i-1], 'o'), 
                 color='#FFFFFF', fontsize=13, horizontalalignment='center')
plt.axhline(y = white_trivial_SSE, color='k', linestyle="--")
plt.show()
plt.clf()
data = pd.DataFrame(list(zip(models_red, SSE_red)), 
               columns =['Model', 'SSE'])

sns.barplot(data=data, x="Model", y="SSE")
sns.set(style="whitegrid")
sns.despine(left=True)
plt.title('Classification SSE - Red Wine')
for i in range(len(SSE_red)+1):
    plt.text(x=i-1, y=100, s=format(SSE_red[i-1], 'o'), 
                 color='#FFFFFF', fontsize=13, horizontalalignment='center')
plt.axhline(y = red_trivial_SSE, color='k', linestyle="--")
plt.show()
plt.clf()


### Misearble Analysis
