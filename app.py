""" ***************************************************************************
# * File Description:                                                         *
# * Workflow for model building                                               *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Get data                                                               *
# * 3. Create train and test set                                              *
# * 4. Classifiers                                                            *
# * 5. Hyper-parameters                                                       *
# * 6. Feature Selection: Removing highly correlated features                 *
# * 7. Tuning a classifier to use with RFECV                                  *
# * 8. Custom pipeline object to use with RFECV                               *
# * 9. Feature Selection: Recursive Feature Selection with Cross Validation   *
# * 10. Performance Curve                                                     *
# * 11. Feature Selection: Recursive Feature Selection                        *
# * 12. Visualizing Selected Features Importance                              *
# * 13. Classifier Tuning and Evaluation                                      *
# * 14. Visualing Results                                                     *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Frank Ceballos <frank.ceballos89@gmail.com>                   *
# * --------------------------------------------------------------------------*
# * DATE CREATED: July 22, 2019                                               *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


###############################################################################
#                                 2. Get data                                 #
###############################################################################
data = pd.read_csv("HTRU1.csv")


###############################################################################
#                        3. Create train and test set                         #
###############################################################################
data_train, data_test = train_test_split(data, test_size = 0.25,
                                         random_state = 10)

# Get test feature matrix and class target
X_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]


###############################################################################
#                          4. Get balanced dataset                            #
###############################################################################
# Get class 0 and class 1
class_0 = data_train[data_train["Class"] == 0]
class_1 = data_train[data_train["Class"] == 1]

# Randomly under sample the majority class
class_0_under = class_0.sample(len(class_1))

# Join class_0 and class_1
balanced_data = pd.concat([class_0_under, class_1], axis = 0)

# Shuffle dataset
balanced_data = balanced_data.sample(frac = 1)

# Get train feature matrix and class target
X_train = balanced_data.iloc[:,:-1]
y_train = balanced_data.iloc[:, -1]


###############################################################################
#                           5. Visualize data set                             #
###############################################################################
# Set graph style
sns.set(font_scale = 0.3)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4', 'axes.grid': False})

# Define color palette
palette = sns.color_palette("RdYlGn", 10)
palette = [palette[0], palette[-1]]

# Make graph
g = sns.pairplot(data = balanced_data, palette  = palette, 
                 plot_kws=dict(s=1, edgecolor= None, linewidth=0.5), hue = "Class")
g.fig.set_size_inches(12,9)
g.savefig("Pair Plot.png", dpi = 1080)


###############################################################################
#                               6. Classifiers                                #
###############################################################################
# Create list of tuples with classifier label and classifier object
classifiers = {}
classifiers.update({"LDA": LinearDiscriminantAnalysis()})
classifiers.update({"QDA": QuadraticDiscriminantAnalysis()})
classifiers.update({"AdaBoost": AdaBoostClassifier()})
classifiers.update({"Bagging": BaggingClassifier()})
classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
classifiers.update({"Random Forest": RandomForestClassifier()})
classifiers.update({"Ridge": RidgeClassifier()})
classifiers.update({"SGD": SGDClassifier()})
classifiers.update({"BNB": BernoulliNB()})
classifiers.update({"GNB": GaussianNB()})
classifiers.update({"KNN": KNeighborsClassifier()})
classifiers.update({"MLP": MLPClassifier()})
classifiers.update({"LSVC": LinearSVC()})
classifiers.update({"NuSVC": NuSVC()})
classifiers.update({"SVC": SVC()})
classifiers.update({"DTC": DecisionTreeClassifier()})
classifiers.update({"ETC": ExtraTreeClassifier()})

# Create dict of decision function labels
DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}


###############################################################################
#                             7. Hyper-parameters                             #
###############################################################################
# Initiate parameter grid
parameters = {}

# Update dict with LDA
parameters.update({"LDA": {"classifier__solver": ["svd"], 
                                         }})

# Update dict with QDA
parameters.update({"QDA": {"classifier__reg_param":[0.01*ii for ii in range(0, 101)], 
                                         }})
# Update dict with AdaBoost
parameters.update({"AdaBoost": { 
                                "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "classifier__n_estimators": [200],
                                "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
                                 }})

# Update dict with Bagging
parameters.update({"Bagging": { 
                                "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "classifier__n_estimators": [200],
                                "classifier__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                "classifier__n_jobs": [-1]
                                }})

# Update dict with Gradient Boosting
parameters.update({"Gradient Boosting": { 
                                        "classifier__learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                                        "classifier__n_estimators": [200],
                                        "classifier__max_depth": [2,3,4,5,6],
                                        "classifier__min_samples_split": [0.001, 0.01, 0.05, 0.10],
                                        "classifier__min_samples_leaf": [0.001, 0.01, 0.05, 0.10],
                                        "classifier__max_features": ["auto", "sqrt", "log2"],
                                        "classifier__subsample": [1]
                                         }})


# Update dict with Extra Trees
parameters.update({"Extra Trees Ensemble": { 
                                            "classifier__n_estimators": [200],
                                            "classifier__class_weight": [None, "balanced"],
                                            "classifier__max_features": ["auto", "sqrt", "log2"],
                                            "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                            "classifier__min_samples_split": [0.001, 0.01, 0.05, 0.10],
                                            "classifier__min_samples_leaf": [0.001, 0.01, 0.05, 0.10],
                                            "classifier__criterion" :["gini", "entropy"]     ,
                                            "classifier__n_jobs": [-1]
                                             }})


# Update dict with Random Forest Parameters
parameters.update({"Random Forest": { 
                                    "classifier__n_estimators": [200],
                                    "classifier__class_weight": [None, "balanced"],
                                    "classifier__max_features": ["auto", "sqrt", "log2"],
                                    "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                    "classifier__min_samples_split": [0.001, 0.01, 0.05, 0.10],
                                    "classifier__min_samples_leaf": [0.001, 0.01, 0.05, 0.10],
                                    "classifier__criterion" :["gini", "entropy"]     ,
                                    "classifier__n_jobs": [-1]
                                     }})

# Update dict with Ridge
parameters.update({"Ridge": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                             }})

# Update dict with SGD Classifier
parameters.update({"SGD": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                            "classifier__penalty": ["l1", "l2"],
                            "classifier__n_jobs": [-1]
                             }})


# Update dict with BernoulliNB Classifier
parameters.update({"BNB": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                             }})

# Update dict with GaussianNB Classifier
parameters.update({"GNB": { 
                            "classifier__var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]
                             }})

# Update dict with K Nearest Neighbors Classifier
parameters.update({"KNN": { 
                            "classifier__n_neighbors": list(range(1,31)),
                            "classifier__p": [1, 2, 3, 4, 5],
                            "classifier__leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                            "classifier__n_jobs": [-1]
                             }})

# Update dict with MLPClassifier
parameters.update({"MLP": { 
                            "classifier__hidden_layer_sizes": [(5), (10), (5,5), (10,10), (10,10,10)],
                            "classifier__activation": ["identity", "logistic", "tanh", "relu"],
                            "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
                            "classifier__max_iter": [100, 200, 300, 500],
                            "classifier__alpha": list(10.0 ** -np.arange(5, 10)),
                             }})

parameters.update({"LSVC": { 
                            "classifier__penalty": ["l2"],
                            "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
                             }})

parameters.update({"NuSVC": { 
                            "classifier__nu": [0.25, 0.50, 0.75],
                            "classifier__kernel": ["linear", "rbf", "poly"],
                            "classifier__degree": [1,2,3,4,5,6],
                             }})

parameters.update({"SVC": { 
                            "classifier__kernel": ["linear", "rbf", "poly"],
                            "classifier__gamma": ["auto"],
                            "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100],
                            "classifier__degree": [1, 2, 3, 4, 5, 6]
                             }})


# Update dict with Decision Tree Classifier
parameters.update({"DTC": { 
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})

# Update dict with Extra Tree Classifier
parameters.update({"ETC": { 
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})


###############################################################################
#                       8. Classifier Tuning and Evaluation                   #
###############################################################################
# Initialize dictionary to store results
results = {}

# Tune and evaluate classifiers
for classifier_label, classifier in classifiers.items():
    # Print message to user
    print(f"Now tuning {classifier_label}.")
    
    # Scale features via Z-score normalization
    scaler = StandardScaler()
    
    # Define steps in pipeline
    steps = [("scaler", scaler), ("classifier", classifier)]
    
    # Initialize Pipeline object
    pipeline = Pipeline(steps = steps)
      
    # Define parameter grid
    param_grid = parameters[classifier_label]
    
    # Initialize GridSearch object
    gscv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = "recall")
                      
    # Fit gscv
    gscv.fit(X_train, np.ravel(y_train))  
    
    # Get best parameters and score
    best_params = gscv.best_params_
    best_score = gscv.best_score_
    
    # Update classifier parameters and define new pipeline with tuned classifier
    tuned_params = {item[12:]: best_params[item] for item in best_params}
    classifier.set_params(**tuned_params)
            
    # Make predictions
    if classifier_label in DECISION_FUNCTIONS:
        y_pred = gscv.decision_function(X_test)
    else:
        y_pred = gscv.predict_proba(X_test)[:,1]
    
    # Get AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    
    # Get f1 score
    y_pred = gscv.predict(X_test)
    f1 = metrics.f1_score(y_test, y_pred)
    
    # Get recall score
    recall = metrics.recall_score(y_test, y_pred)

    # Get precision score
    precision = metrics.precision_score(y_test, y_pred)
    
    # False Positive rate
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fp_rate = (fp)/(fp + tn)

    # Save results
    result = {"Classifier": gscv,
              "Best Parameters": best_params,
              "Training Recall Score": best_score,
              "Test Recall Score": recall,
              "Test Precision Score": precision,
              "Test F1 Score": f1,
              "False Positive Rate": fp_rate,
              "Test AUC": auc}
    
    results.update({classifier_label: result})


###############################################################################
#                              9. Visualing Results                           #
###############################################################################
# Helper function to show 
def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
        
# Recall scores plot
recall_scores = {
              "Classifier": [],
              "Recall Score": [],
              }

# Get recall scores into dictionary
for classifier_label in results:  
    recall_scores.update({"Classifier": [classifier_label] + recall_scores["Classifier"],
                       "Recall Score": [results[classifier_label]["Test Recall Score"]] + recall_scores["Recall Score"]
                       })

# Dictionary to PandasDataFrame
recall_scores = pd.DataFrame(recall_scores)

# Sort dataframe
recall_scores = recall_scores.sort_values(by = ["Recall Score"], ascending = False)

# Convert decimals to percentages
recall_scores[recall_scores.select_dtypes(include=['number']).columns] *= 100

# Set graph style
sns.set(font_scale = 1.75)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Colors
palette = sns.color_palette("YlOrRd", 20)[::-1]

# Set figure size and create barplot
f, ax = plt.subplots(figsize=(12, 9))

g = sns.barplot(x="Recall Score", y="Classifier", palette = palette,
            data = recall_scores)

# Generate a bolded horizontal line at y = 0
ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)

# Turn frame off
ax.set_frame_on(False)

# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig("Recall Scores Bar Plot.png", dpi = 1080)

show_values_on_bars(g, h_v="h", space=0.4)



# False positive rate plot
fpr_scores = {
              "Classifier": [],
              "False Positive Rate": [],
              }

# Get fpr into dictionary
for classifier_label in results:
    fpr_scores.update({"Classifier": [classifier_label] + fpr_scores["Classifier"],
                       "False Positive Rate": [results[classifier_label]["False Positive Rate"]] + fpr_scores["False Positive Rate"]
                       })
    

# Dictionary to PandasDataFrame
fpr_scores = pd.DataFrame(fpr_scores)

# Sort dataframe
fpr_scores = fpr_scores.sort_values(by = ["False Positive Rate"])

# Set graph style
sns.set(font_scale = 1.75)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Colors
palette = sns.color_palette("YlGn", 20)[::-1]


# Set figure size and create barplot
f, ax = plt.subplots(figsize=(12, 9))

sns.barplot(x="False Positive Rate", y="Classifier", palette = palette,
            data = fpr_scores)


# Generate a bolded horizontal line at y = 0
ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)

# Turn frame off
ax.set_frame_on(False)

# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig("False Positive Rate Bar Plot.png", dpi = 1080)