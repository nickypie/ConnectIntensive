# Getting Started
This repo is a collection of [Jupyter Notebooks](http://jupyter.org/) to accompany the Udacity Connect Intensive [Machine Learning Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009). The code is written for Python 2.7, but should be (mostly) compatible with Python 3.x.

## Installing Python and Jupyter Notebook
If you haven't already done so, you'll need to [download and install Python 2.7](https://www.python.org/downloads/). If using Mac OS X, you may want to use [Homebrew](http://brew.sh/) as a package manager, [following these instructions](http://docs.python-guide.org/en/latest/starting/install/osx/) to install Python 2.7 or Python 3. You can also use [Anaconda](https://www.continuum.io/downloads) as a package manager. Then, you can [follow these instructions](http://jupyter.readthedocs.io/en/latest/install.html) to install Jupyter notebook. [These instructions](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) explain how to install both Python 2 and Python 3 kernels.

## Fork and Clone this Repo
You can [follow these instructions](https://help.github.com/articles/fork-a-repo/) to create a fork of the ConnectIntensive repo, and clone it to your local machine. Once you've done so, you can navigate to your local clone of the ConnectIntensive repo and [follow these instructions](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) to run the Jupyter Notebook App.

## Required Libraries and Packages
The required packages and libraries vary in each of these Jupyter Notebooks. The most commonly used ones are listed below:
  - [matplotlib](http://matplotlib.org/)
  - [numpy](http://www.numpy.org/)
  - [pandas](http://pandas.pydata.org/)
  - [sklearn](http://scikit-learn.org/stable/)
  
Each Lesson Notebook lists its own specific prerequisites along with the objectives.


# Lesson Notebooks
Most lesson notebooks have a corresponding solutions notebook with the outputs of each cell shown. For example, the notebook `solutions-01.ipynb` displays the output and shows the solutions to the exercises from `lesson-01.ipynb`.
  - `lesson-00.ipynb` : Hello Jupyter Notebook!
    - A "hello world" notebook to introduce the Jupyter IDE
    - Introduces [import statements](https://docs.python.org/2/tutorial/modules.html) for commonly-used modules and packages
  - `lesson-01.ipynb` : An intro to Statistical Analysis using `pandas`
    - Introduces [the `Series` and `DataFrame` objects](http://pandas.pydata.org/pandas-docs/stable/dsintro.html) in `pandas`
    - Defines [categorical variables](http://pandas.pydata.org/pandas-docs/stable/categorical.html)
    - Covers basic [descriptive statistics](https://en.wikipedia.org/wiki/Descriptive_statistics): mean, median, min/max
    - Label-based `.loc` and index-based location `.iloc` in `pandas`
    - [Boolean indexing](http://pandas.pydata.org/pandas-docs/stable/indexing.html), how to slice a `DataFrame` in `pandas`
    - Exercises in exploratory data analysis, emphasizing [`groupby`](http://pandas.pydata.org/pandas-docs/stable/groupby.html) and [`plot`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html)
  - `lesson-02.ipynb` : Working with the Enron Data Set
    - Covers [the `pickle` module](https://docs.python.org/2/library/pickle.html) for saving objects
    - [Magic commands](http://ipython.readthedocs.io/en/stable/interactive/magics.html) in Jupyter notebooks
    - Use of [the `stack` and `unstack` functions](http://pandas.pydata.org/pandas-docs/stable/reshaping.html) in `pandas`
    - Exercises in exploratory data analysis on the Enron data set
  - `lesson-03-part-01.ipynb` : Building and Evaluating Models with `sklearn` (part 1)
    - Perform exploratory data analysis on a dataset
    - Tidy a data set so that it will be compatible with the `sklearn` library
      - Use [the `pandas.get_dummies()` method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) to convert categorical variables to dummy or indicator variables.
      - <a href="https://en.wikipedia.org/wiki/Imputation_(statistics)">Impute</a> missing values to ensure variables are numeric.
  - `lesson-03-part-02.ipynb` : Building and Evaluating Models with `sklearn` (part 2)
    - Make [decision tree classifiers](http://scikit-learn.org/stable/modules/tree.html) on the tidied dataset from part 01
    - Compute [the accuracy score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) of a model on both the training and validation (testing) data
    - Adjust [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_optimization) to see the effects on model accuracy
    - Use [`export_graphviz`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html) to visualize decision trees.
    - Introduce the [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
  - `lesson-04-part-01.ipynb` : Bayes NLP Mini-Project
    - Understand how [Bayes' Rule](https://en.wikipedia.org/wiki/Bayes'_rule) derives from conditional probability
    - Write methods, applying Bayesian learning to simple word-prediction tasks
    - Practice with [python string methods](https://docs.python.org/2/library/string.html), e.g. `str.split()`, and [python dictionaries](https://docs.python.org/2/library/stdtypes.html)
  - `lesson-05.ipynb` : Classification with Support Vector Machines
    - Introduces additional plotting functionality in [`matplotlib.pyplot`](http://matplotlib.org/api/pyplot_api.html)
      - [Boxplots](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.boxplot) for depicting interquartile range (IQR), median, max, min, outliers
      - [Scatterplots](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter) for 2-D representation of two features.
    - Introduction to [Support Vector Machines in `sklearn`](http://scikit-learn.org/stable/modules/svm.html)
      - An introduction to [kernels](https://www.quora.com/What-are-Kernels-in-Machine-Learning-and-SVM)
      - [Hard-margin](https://en.wikipedia.org/wiki/Support_vector_machine#Hard-margin) versus [soft-margin](https://en.wikipedia.org/wiki/Support_vector_machine#Soft-margin) SVMs
      - Overview of `SVC` hyperparameters: [`C`](http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel), `gamma`, `degree`, etc.
    - Visualize decision boundaries resulting from the different kernels
    - Practice with [the `GridSearchCV()` method](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
  - `lesson-06-part-01.ipynb` : Clustering Mini-Project
    - Perform [k-means clustering](http://scikit-learn.org/stable/modules/clustering.html#k-means) on the Enron Data Set.
    - Visualize different clusters that form before and after feature scaling.
    - Plot decision boundaries that arise from k-means clustering using two features.
  - `lesson-06-part-02.ipynb` : PCA Mini-Project
    - Perform [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) on a large set of features.
    - Recognize differences between [`train_test_split()`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and [`StratifiedShuffleSplit()`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html).
    - Introduce the `class_weight` parameter for `SVC()`.
    - Visualize the eigenfaces (orthonormal basis of components) that result from PCA.
    
# Additional Resources
I find that learning Python from Jupyter Notebooks is addictive. Here are some other great resources.
  - [Thomas Corcoran's Connect Repo](https://github.com/tccorcoran/Connect): More notebooks prepared by another talented MLND Session Lead
  - [Brandon Rhodes' PyCon 2015 Pandas Tutorial](https://github.com/brandon-rhodes/pycon-pandas-tutorial): One of my favorite introductions to `pandas` with an accompanying [video lecture](https://www.youtube.com/watch?v=5JnMutdy6Fw).
  - [Jake VanderPlas' Scikit-learn Tutorial](https://github.com/jakevdp/sklearn_tutorial): An introduction to `sklearn`, also with an accompanying [video lecture](https://www.youtube.com/watch?v=L7R4HUQ-eQ0)
  - [Kevin Markham's Machine Learning with Text in Scikit-learn Tutorial](https://www.youtube.com/watch?v=WHocRqT-KkU): If you want to get started with NLP using `sklearn`, Kevin's tutorial is a great introduction ([video lecture here](https://www.youtube.com/watch?v=WHocRqT-KkU)).
    
