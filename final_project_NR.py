# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
#data_path = './winequalityN.csv'
#HI 
data_path = r'winequalityN.csv'
wine = pd.read_csv(data_path)

'''Some things to consider from the paper...
SVM relative importance plots for attributes?
“We will adopt a regression approach, which preserves the order of the preferences.
For instance, if the true grade is 3, then a model that predicts 4 is better than one that predicts 7.”'''

# %% 
## Defining functions
from sklearn.feature_selection import SelectKBest

def FeatureSelection(k, df, x_train, y_train, x_test, y_test):
    s = SelectKBest(k=k).fit(x_train, y_train)
    mask = s.get_support(True)
    selected_features = df.columns[mask].tolist()
    train_selected = SelectKBest(k=k).fit_transform(x_train, y_train)
    test_selected = SelectKBest(k=k).fit_transform(x_test, y_test)
    return train_selected, test_selected, selected_features

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
# dropping 'outliers', i.e., records w/ 'quality' values of 3 or 9
white = white[(white.quality != 3) & (white.quality != 9)]
red = red[(red.quality != 3) & (red.quality != 9)]




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
# but it can easily be adjusted for preliminary models with test_size=0.33
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# labels for confusion matrices
# TODO: fix labels, remove 3 and 9 - N
white_labels = [4, 5, 6, 7, 8]
red_labels = [4, 5, 6, 7, 8]
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

# TODO: standardized to a zero mean and one standard deviation for input attr - N
# standardizing feature values
scaler = StandardScaler()
white_train_x = scaler.fit_transform(white_train_x)
white_test_x = scaler.fit_transform(white_test_x)
red_train_x = scaler.fit_transform(red_train_x)
red_test_x = scaler.fit_transform(red_test_x)

# Models and metrics lists for later plotting/comparison
models_white = []
accuracy_white = []
SSE_white = []
models_red = []
accuracy_red = []
SSE_red = []

# TODO: 2D representation of classifier splitting data - J

# %%
#perplexity = 50 for white test data perplexity = 20 for red test data
#I might have to play around a little bit more
def two_dimensional_representation(x_data,y_data,title="t-SNE wine",perplexity = 50):
    tsne = TSNE(verbose=1, perplexity=perplexity, random_state = 42)
    X_embedded_data = tsne.fit_transform(x_data)

    # sns settings
    sns.set(rc={'figure.figsize':(10,10)})

    # colors
    palette = sns.color_palette("bright", len(set(y_data)))

    # plot
    sns.scatterplot(X_embedded_data[:,0], X_embedded_data[:,1], hue = y_data, palette=palette)

    plt.title(title)
    # plt.savefig("plots/t-sne_wine.png")
    plt.show()

two_dimensional_representation(red_test_x,red_test_y,"Red actual",20)
two_dimensional_representation(white_test_x,white_test_y,"White actual",50)
#%%

# defining SSE
def SSE(actual, pred):
    s = 0
    for i in range(len(actual)):
        s += abs(actual[i]-pred[i])**2
    return s

#Prints aout SSE and Accuracy Data and prints out graph
#Example
#svm_function(white_train_x, white_train_y, white_test_x, white_test_y)
#data_analyze("White", white_svm, "SVM")
def data_analyze(wine_color,classifier,classifier_name):
    labels = ""
    if(wine_color=="Red"):
        y_pred = classifier.predict(red_test_x)
        test_y = red_test_y
        labels = red_labels
        acc = accuracy_score(test_y, y_pred)
        accuracy_red.append(acc)
        SSE_red.append(SSE(test_y, y_pred))
    elif(wine_color=="White"):
        y_pred = classifier.predict(white_test_x)
        test_y = white_test_y
        labels = white_labels
        acc = accuracy_score(test_y, y_pred)
        accuracy_white.append(acc)
        SSE_white.append(SSE(test_y, y_pred))
    else:
        print("Bad Wine Color")
        raise
    print (f" Accuracy for {classifier_name} on {wine_color} dataset is {acc}")
    print (f" SSE for {classifier_name} on {wine_color} dataset is {SSE}")


    data = confusion_matrix(test_y, y_pred, labels = labels)
    df_cm = pd.DataFrame(data, columns = labels, index = labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    cm = sns.heatmap(df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
    cm.tick_params(left = False, bottom = False)
    cm.set_title(f'{classifier_name} - {wine_color} Wine')
    plt.show()
    plt.clf()

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

#%%
#import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


# %%
## Replicate Author's SVM
# Support vector machines
# TODO: Mathew: make function and try to replicate - M
from sklearn import svm
from sklearn.model_selection import GridSearchCV
np.set_printoptions(precision=4)
def Authors_SVM(x_train, y_train, x_test, y_test):
    # using gamma values that the authors found were the best
    white_gamma = 2**1.55  # 2.928
    red_gamma = 2**0.19  # 1.14
    gamma = np.logspace(-3, 6, 20, 2)
    parameters = {'C':range(1,20), 'gamma':gamma}
    best_w = {'C': 2, 'gamma': 0.6951927961775606}
    best_r = {'C': 1, 'gamma': 0.07847599703514611}
    author_params = {'C':[3], 'gamma':[white_gamma, red_gamma]}
    svm_clf = svm.SVR(kernel='rbf')


    clf = GridSearchCV(svm_clf, parameters, n_jobs=1, verbose=True, cv=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred = np.rint(y_pred)  # round predictions to nearest integer for classification

    print('best parameters: {}', clf.best_params_)
    print('best score: {}', clf.best_score_)
    print("MAE: {}", mean_absolute_error(y_test, y_pred))

    # print confusion matrix
    conf = confusion_matrix(y_test, y_pred)
    print(conf)
    
    # get precision scores
    prec_w = precision_score(y_test, y_pred, average=None, zero_division=0)
    print(prec_w)

    # from sklearn.metrics import classification_report
    

Authors_SVM(white_train_x, white_train_y, white_test_x, white_test_y)
Authors_SVM(red_train_x, red_train_y, red_test_x, red_test_y)





#%%
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





## Unsupervised Learning

# %%
# # Clustering
# from sklearn.cluster import KMeans
# TODO: make function - Jack
from sklearn.cluster import KMeans
def K_means(x_train, y_train, x_test, y_test, title = "Wine"):
    distortions = []
    for k in range(1,11):
        kmeans = KMeans(n_clusters=k, verbose=False, random_state=42)
        kmeans.fit(x_train, y_train)
        distortions.append(kmeans.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title(title)
    plt.show()

    # print confusion matrix
    #conf = confusion_matrix(y_test, y_pred)
    #print(conf)
    
    # get precision scores
    #prec_w = precision_score(y_test, y_pred, average=None, zero_division=0)
    #print(prec_w)

K_means(white_train_x, white_train_y, white_test_x, white_test_y,"white wine")
K_means(red_train_x, red_train_y, red_test_x, red_test_y,"red wine")

# %%
# Nearest Neighbors, the unsupervised version doesn't allow for classification
# TODO: Mathew: make function - M
from sklearn.neighbors import RadiusNeighborsClassifier
def RNC(x_train, y_train, x_test, y_test):
    parameters = {
        'weights': ['uniform', 'distance'],
        'radius': np.arange(1.0, 11.0, 0.5),
        'n_jobs': [1],
        'outlier_label':['most_frequent'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
        }
    best_w = {'algorithm': 'ball_tree', 'n_jobs': 1, 'outlier_label': 'most_frequent', 'radius': 2.5, 'weights': 'distance'}
    best_r = {'algorithm': 'ball_tree', 'n_jobs': 1, 'outlier_label': 'most_frequent', 'radius': 2.5, 'weights': 'distance'}
    model = RadiusNeighborsClassifier()
    clf = GridSearchCV(model, parameters, n_jobs=1, verbose=True, cv=3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred = np.rint(y_pred)  # round predictions to nearest integer for classification

    print('best parameters: {}', clf.best_params_)
    print('best score: {}', clf.best_score_)
    print("MAE: {}", mean_absolute_error(y_test, y_pred))

    # print confusion matrix
    conf = confusion_matrix(y_test, y_pred)
    print(conf)
    
    # get precision scores
    prec_w = precision_score(y_test, y_pred, average=None, zero_division=0)
    print(prec_w)

    # TODO: Mathew: once we have the best parameters, maybe then we should do a special run and analysis of that model?
    
RNC(white_train_x, white_train_y, white_test_x, white_test_y)
RNC(red_train_x, red_train_y, red_test_x, red_test_y)




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
def gnb()
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
def svm_function(x_train, y_train, x_test, y_test):
    rbf = svm.SVC(kernel = 'rbf', random_state = 42)
    rbf.fit(x_train,y_train)
    return rbf

white_svm = svm_function(white_train_x, white_train_y, white_test_x, white_test_y)
data_analyze("White", white_svm, "SVM")

red_svm = svm_function(red_train_x, red_train_y, red_test_x, red_test_y)
data_analyze("Red", red_svm, "SVM")




# %%
# # Neural network - might not have great performance
# TODO: Mathew: make function - M
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
def MLP(x_train, y_train, x_test, y_test):
    parameters = {
        'activation': ['logistic', 'identity', 'tanh', 'relu'],
        'alpha': [0.01, 0.001, 0.0001, 0.00001],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'solver': ['lbfgs', 'sgd', 'adam'], 
        'random_state':[42],
        'max_iter': [1000],

        }
    best_w = {'activation': 'tanh', 'alpha': 0.01, 'learning_rate': 'constant', 'random_state': 42}
    best_r = {'activation': 'relu', 'alpha': 0.01, 'learning_rate': 'constant', 'random_state': 42}
    model = MLPClassifier()
    clf = GridSearchCV(model, parameters, n_jobs=1, verbose=True, cv=3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred = np.rint(y_pred)  # round predictions to nearest integer for classification

    print('best parameters: {}', clf.best_params_)
    print('best score: {}', clf.best_score_)
    print("MAE: {}", mean_absolute_error(y_test, y_pred))

    # print confusion matrix
    conf = confusion_matrix(y_test, y_pred)
    print(conf)
    
    # get precision scores
    prec_w = precision_score(y_test, y_pred, average=None, zero_division=0)
    print(prec_w)
    

MLP(white_train_x, white_train_y, white_test_x, white_test_y)
MLP(red_train_x, red_train_y, red_test_x, red_test_y)



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
