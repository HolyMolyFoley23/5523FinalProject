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

def confusion(test, pred, labels, title):
    d = confusion_matrix(test, pred, labels = labels)
    df_cm = pd.DataFrame(d, columns = labels, index = labels)
    df_cm.columns.name = 'Predicted'
    df_cm.index.name = 'Actual'
    cm = sns.heatmap(df_cm, cmap = 'Blues', linewidths = 0.1, annot=True, fmt = 'd')
    cm.tick_params(left = False, bottom = False)
    cm.set_title(title)
    plt.show()
    plt.clf()

def plot_metrics(df_scores, models, color):
    sns.set(style="whitegrid")
    sns.despine(left=True)
    for column in df_scores.columns:
         plot = sns.barplot(y = df_scores[column], x = models)
         plt.title(f'Classification {column} - {color} Wine dataset')
         for p in plot.patches:
             plot.annotate(format(p.get_height(), '0.5f'),
                           (p.get_x() + p.get_width()/2, p.get_height()), 
                            ha = 'center', va = 'center',
                            xytext = (0,-20), 
                            textcoords = 'offset points', color = 'white')
         plt.show()
         plt.clf()
    
# %%
## Classifier functions

def do_trivial(train_x, train_y, test_x, test_y, metrics_list, metrics_names, color):
    scores = []
    pred = np.full(test_y.shape, stats.mode(train_y)[0])
    for i in range(len(metrics_list)):
        scores.append(metrics_list[i](test_y, pred)) 
        print(f" {metrics_names[i]} for trivial classifier on {color.lower()} dataset is {scores[i]}.")
    confusion(test_y, pred, labels=labels,
          title=f'Trivial Classifer - {color} Wine')
    return pred, scores

def do_knn(train_x, train_y, test_x, test_y, params, metrics_list, metrics_names, color):
    scores = []
    k = KNeighborsClassifier(n_neighbors = params['n_neighbors'],
                           weights = params['weights'])
    k.fit(train_x, train_y)
    pred = k.predict(test_x)
    for i in range(len(metrics_list)):
        scores.append(metrics_list[i](test_y, pred)) 
        print(f" {metrics_names[i]} for {params['n_neighbors']}-NN classifier on {color.lower()} dataset is {scores[i]}.")
    confusion(test_y, pred, labels=labels,
          title=f"{params['n_neighbors']}-NN - {color} Wine")
    return pred, scores

def do_tree(train_x, train_y, test_x, test_y, params, metrics_list, metrics_names, color):
    scores = []
    dt = tree.DecisionTreeClassifier(max_depth = params['max_depth'],
                                      max_leaf_nodes = params['max_leaf_nodes'],
                                      criterion = params['criterion'],
                                      random_state = 42)
    d = dt.fit(train_x, train_y)
    pred = d.predict(test_x)
    for i in range(len(metrics_list)):
        scores.append(metrics_list[i](test_y, pred)) 
        print(f" {metrics_names[i]} for decision tree classifier on {color.lower()} dataset is {scores[i]}.")
    confusion(test_y, pred, labels=labels,
          title=f"Decision tree classifier - {color} Wine")
    return pred, scores
              
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

bins = [0,1,2,3,4,5,6,7,8,9,10,11]

# white wine
fig, ax = plt.subplots(1,1)
ax.hist(white['quality'], bins=bins, align='left')
ax.set_title('Histogram of White Wine Quality')
ax.set_xlabel('Quality')
ax.set_ylabel('Count')
sns.set(style="whitegrid")
rects = ax.patches
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.5, '{:.0f}'.format(height), ha='center', va = 'bottom')
plt.show()
plt.clf()

# red wine
fig, ax = plt.subplots(1,1)
ax.hist(red['quality'], bins=bins, align='left')
ax.set_title('Histogram of Red Wine Quality')
ax.set_xlabel('Quality')
ax.set_ylabel('Count')
sns.set(style="whitegrid")
rects = ax.patches
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.5, '{:.0f}'.format(height), ha='center', va = 'bottom')
plt.show()
plt.clf()
              
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
labels = [4, 5, 6, 7, 8]
# Histograms
# white wine
fig, ax = plt.subplots(1,1)
ax.hist(white['quality'], bins=labels, align='left')
ax.set_title('Histogram of White Wine Quality')
ax.set_xlabel('Quality')
ax.set_ylabel('Count')
sns.set(style="whitegrid")
rects = ax.patches
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.5, '{:.0f}'.format(height), ha='center', va = 'bottom')
plt.show()
plt.clf()
# red wine
fig, ax = plt.subplots(1,1)
ax.hist(red['quality'], bins=labels, align='left')
ax.set_title('Histogram of Red Wine Quality')
ax.set_xlabel('Quality')
ax.set_ylabel('Count')
sns.set(style="whitegrid")
rects = ax.patches
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.5, '{:.0f}'.format(height), ha='center', va = 'bottom')
plt.show()
plt.clf()
# appears that quality of both red and white wine follows roughly a normal distribution

# Correlation maps
# white wine
plt.figure(figsize=(10,6))
corr_white = sns.heatmap(white.corr(), annot=True, cmap='Blues')
corr_white.set_title('Correlation Map - White Wine')
plt.show()
plt.clf()
# red wine
plt.figure(figsize=(10,6))
corr_red = sns.heatmap(red.corr(), annot=True, cmap='Blues')
corr_red.set_title('Correlation Map - Red Wine')
plt.show()
plt.clf()
# alcohol content has highest correlation with quality for both red and white wine

# %%
### Train/Test Split and other Utilities
# Paper did a 2/3 1/3 split

# It eventually won't matter when we do k-fold cross validation
# but it can easily be adjusted for preliminary models with test_size=0.33
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# labels for confusion matrices
# TODO: fix labels, remove 3 and 9 - N

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
              
# creating index of features
features = white_train_x.columns

# TODO: standardized to a zero mean and one standard deviation for input attr - N
# standardizing feature values
scaler = StandardScaler()
white_train_x = scaler.fit_transform(white_train_x)
white_test_x = scaler.fit_transform(white_test_x)
red_train_x = scaler.fit_transform(red_train_x)
red_test_x = scaler.fit_transform(red_test_x)

# Models and metrics lists for later plotting/comparison
models = []
scores_white = []  
scores_red = []
models_white = []
accuracy_white = []
SSE_white = []
models_red = []
accuracy_red = []
SSE_red = []

# TODO: 2D representation of classifier splitting data - J

#%%
# Experimenting with feature selection
def diff(l1, l2):
    return (list(list(set(l1)-set(l2))+ list(set(l2)-set(l1))))

l_red = []
l_white = []
sel_features_inorder_white = []
sel_features_inorder_red = []
for i in range(1, 12):
    white_train, white_test, white_selected_features = FeatureSelection(i, white_x, white_train_x, white_train_y, white_test_x, white_test_y)
    l_white.append(white_selected_features)
    red_train, red_test, red_selected_features = FeatureSelection(i, red_x, red_train_x, red_train_y, red_test_x, red_test_y)
    l_red.append(red_selected_features)
    if i == 1:
        sel_features_inorder_white.append(l_white[0])
        sel_features_inorder_red.append(l_red[0])
    else:
        sel_features_inorder_white.append(diff(l_white[i-1], l_white[i-2]))
        sel_features_inorder_red.append(diff(l_red[i-1], l_red[i-2]))
        
print(f'White selected features, in order: {sel_features_inorder_white}')
print(f'Red selected features, in order: {sel_features_inorder_red}')
              
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
    labels = [4,5,6,7,8]
    if(wine_color=="Red"):
        y_pred = classifier.predict(red_test_x)
        test_y = red_test_y
        acc = accuracy_score(test_y, y_pred)
        SSE_data = SSE(test_y, y_pred)
        SSE_red.append(SSE_data)
    elif(wine_color=="White"):
        y_pred = classifier.predict(white_test_x)
        test_y = white_test_y
        acc = accuracy_score(test_y, y_pred)
        accuracy_white.append(acc)
        SSE_data = SSE(test_y, y_pred)
        SSE_white.append(SSE_data)
    else:
        print("Bad Wine Color")
        #raise
    print (f" Accuracy for {classifier_name} on {wine_color} dataset is {acc}")
    print (f" SSE for {classifier_name} on {wine_color} dataset is {SSE_data}")
    title = f"{classifier_name} - {wine_color} Wine"
    confusion(test_y, y_pred, labels,title)


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

# unsure if this is the way we're going to go, but including these for now. will need to add additional metrics.
metrics_list = [accuracy_score, SSE] #classifiers will apply each of these metrics 
metrics_names = ['Accuracy', 'SSE'] #for plotting

# %%
# trivial classifier 
# predicts the mode (6 for white records, 5 for red) for all records
# this will serve as baseline for performance
# TODO: make function - N
from scipy import stats

white_trivial_pred, white_trivial_scores = do_trivial(white_train_x, white_train_y, white_test_x, white_test_y, metrics_list, metrics_names, 'white')
red_trivial_pred, red_trivial_scores = do_trivial(red_train_x, red_train_y, red_test_x, red_test_y, metrics_list, metrics_names, 'red')

models.append('Trivial')
scores_white.append(white_trivial_scores)
scores_red.append(red_trivial_scores)

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
oneVsRestAnalysis(model, white_train_x, white_train_y, white_test_x, white_test_y, labels)

from sklearn.linear_model import RidgeClassifier as RC
print('RidgeClassifier')
model = RC(random_state=0)
oneVsRestAnalysis(model, white_train_x, white_train_y, white_test_x, white_test_y, labels)
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
from sklearn.neighbors import RadiusNeighborsClassifier
def RNC(x_train, y_train, x_test, y_test, ds_type=None):
    grid_search = {
        'weights': ['uniform', 'distance'],
        'radius': np.arange(1.0, 11.0, 0.5),
        'n_jobs': [1],
        'outlier_label':['most_frequent'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
        }
    best_w = {'algorithm': 'ball_tree', 'n_jobs': 1, 'outlier_label': 'most_frequent', 'radius': 2.5, 'weights': 'distance'}
    best_r = {'algorithm': 'ball_tree', 'n_jobs': 1, 'outlier_label': 'most_frequent', 'radius': 2.5, 'weights': 'distance'}
    param=None
    if ds_type == 'white':
        param = best_w
    elif ds_type == 'red':
        param = best_r
    else:
        param = grid_search
    model = RadiusNeighborsClassifier()
    clf = GridSearchCV(model, param, n_jobs=1, verbose=True, cv=3)
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
    
RNC(white_train_x, white_train_y, white_test_x, white_test_y, 'white')
RNC(red_train_x, red_train_y, red_test_x, red_test_y, 'red')




## Supervised Learning
# %%
# Decision tree
from sklearn.model_selection import RandomizedSearchCV
              
from sklearn import tree

# feature selection
# testing number of features yielded n=5 as ideal number of features for red; using all features was ideal for white
red_train_selected, red_test_selected, red_selected_features = FeatureSelection(5, red_x, red_train_x, red_train_y, red_test_x, red_test_y)              
              
parameters = {'max_depth':range(1,1000), 'criterion' :['gini', 'entropy'],
              'max_leaf_nodes':range(1,1000)}
dt = tree.DecisionTreeClassifier(random_state = 42)
clf = RandomizedSearchCV(dt, parameters, n_jobs=1, n_iter=10, verbose=True, random_state=42, cv=3)

clf.fit(white_train_x, white_train_y)
white_params = clf.best_params_

clf.fit(red_train_selected, red_train_y)
red_params = clf.best_params_

white_tree_pred, white_tree_scores = do_tree(white_train_x, white_train_y, white_test_x, white_test_y, white_params, metrics_list, metrics_names, 'White')
red_tree_pred, red_tree_scores = do_tree(red_train_selected, red_train_y, red_test_selected, red_test_y, red_params, metrics_list, metrics_names, 'Red')

models.append('DT')
scores_white.append(white_tree_scores)
scores_red.append(red_tree_scores)






# %%
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# TODO: make function - Jack
def gaussian_nb(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    return gnb 

white_gnb = gaussian_nb(white_train_x, white_train_y, white_test_x, white_test_y)
data_analyze("White", white_gnb  ,"GNB")

red_gnb = gaussian_nb(red_train_x, red_train_y, red_test_x, red_test_y)
data_analyze("Red", red_gnb, "GNB")





#%%
# K-nearest neighbor
              
# feature selection
# testing feature selection produced best results for using all features for both datasets   
              
parameters = {'n_neighbors':range(1,20), 'weights':['uniform', 'distance']}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, scoring='precision_micro', n_jobs=1, verbose=True, cv=3)

clf.fit(white_train_x, white_train_y)
white_params = clf.best_params_

clf.fit(red_train_x, red_train_y)
red_params = clf.best_params_

white_knn_pred, white_knn_scores = do_knn(white_train_x, white_train_y, white_test_x, white_test_y, white_params, metrics_list, metrics_names, 'white')
red_knn_pred, red_knn_scores = do_knn(red_train_x, red_train_y, red_test_x, red_test_y, red_params, metrics_list, metrics_names, 'red')

models.append('KNN')
scores_white.append(white_knn_scores)
scores_red.append(red_knn_scores)





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
# Neural network - might not have great performance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
def MLP_Classifier(x_train, y_train, x_test, y_test, ds_type=None):
    grid_search = {
        'activation': ['logistic', 'identity', 'tanh', 'relu'],
        'alpha': [0.01, 0.001, 0.0001, 0.00001],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'solver': ['lbfgs', 'sgd', 'adam'], 
        'random_state':[42],
        'max_iter': [1000],
        }
    best_w = {'activation': ['logistic'], 'alpha': [1e-05], 'learning_rate': ['constant'], 'solver': ['lbfgs'], 'random_state': [42], 'max_iter': [100000]}
    best_r = {'activation': ['logistic'], 'alpha': [0.0001], 'learning_rate': ['constant'], 'solver': ['adam'], 'random_state': [42], 'max_iter': [100000]}
    model = MLPClassifier()
    param = None
    if ds_type == 'white': 
        param = best_w
    elif ds_type == 'red':
        param = best_r
    else:
        param = grid_search
    clf = GridSearchCV(model, param, n_jobs=1, verbose=True, cv=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print('best parameters: {}', clf.best_params_)
    print('best score: {}', clf.best_score_)
    print("MAE: {}", mean_absolute_error(y_test, y_pred))

    # print confusion matrix
    conf = confusion_matrix(y_test, y_pred)
    print(conf)
    
    # get precision scores
    prec_w = precision_score(y_test, y_pred, average=None, zero_division=0)
    print(prec_w)
    
MLP_Classifier(white_train_x, white_train_y, white_test_x, white_test_y, 'white')
MLP_Classifier(red_train_x, red_train_y, red_test_x, red_test_y, 'red')

def MLP_Regressor(x_train, y_train, x_test, y_test, ds_type=None):
    grid_search = {
        'activation': ['logistic', 'identity', 'tanh', 'relu'],
        'alpha': [0.01, 0.001, 0.0001, 0.00001],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'solver': ['lbfgs', 'sgd', 'adam'], 
        'random_state':[42],
        'max_iter': [500],
        }
    best_w = {'activation': ['relu'], 'alpha': [0.01], 'learning_rate': ['constant'], 'random_state': [42], 'solver': ['sgd'], 'max_iter': [100000]}
    best_r = {'activation': ['tanh'], 'alpha': [0.01], 'learning_rate': ['constant'], 'random_state': [42], 'solver': ['adam'], 'max_iter': [100000]}
    model = MLPRegressor()
    param = None
    if ds_type == 'white': 
        param = best_w
    elif ds_type == 'red':
        param = best_r
    else:
        param = grid_search
    clf = GridSearchCV(model, param, n_jobs=1, verbose=True, cv=5)
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

MLP_Regressor(white_train_x, white_train_y, white_test_x, white_test_y, 'white')
MLP_Regressor(red_train_x, red_train_y, red_test_x, red_test_y, 'red')




# ### Model evaluation/tuning
# %%
# ## feature selection
# TODO: make function, maybe use PCA? - N
# %%
#from sklearn.decomposition import PCA

#pca = PCA(n_components = 2, random_state = 42)
#white_reduced = pca.fit_transform(white_train_x)
#white_pca = pd.DataFrame(pca.components_,columns=features,index = ['PC-1','PC-2'])
#print(white_pca)

#red_reduced = pca.fit_transform(red_train_x)
#red_pca = pd.DataFrame(pca.components_,columns=features,index = ['PC-1','PC-2'])
#print(red_pca)

## ChooseKBest feature selection - we should play around with this
#no_features = 3
#white_train_selected, white_test_selected, white_selected_features = FeatureSelection(no_features, white_x, white_train_x, white_train_y, white_test_x, white_test_y)
#red_train_selected, red_test_selected, red_selected_features = FeatureSelection(no_features, red_x, red_train_x, red_train_y, red_test_x, red_test_y)


# %%
# Plotting performance
# bar graphs to compare performance of classifiers

df_scores_white = pd.DataFrame(scores_white, columns = metrics_names, index = models)
df_scores_red = pd.DataFrame(scores_red, columns = metrics_names, index = models)

plot_metrics(df_scores_white, models, 'White')
plot_metrics(df_scores_red, models, 'Red')



### Misearble Analysis
