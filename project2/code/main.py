#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_stump import RandomStumpInfoGain
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    for k in [1,3,10]:
        knn = KNN(k)
        knn.fit(X,y)
        y_hat_train = knn.predict(X)
        error_train = 1- np.mean(y_hat_train == y)
        y_hat_test = knn.predict(X_test)
        error_test = 1- np.mean(y_hat_test == y_test)
        print(f"The training error for KNN with k = {k} is {np.round(error_train,4)*100:.1f}%")
        print(f"The testing error for KNN with k = {k} is {np.round(error_test,4)*100:.1f}%")
        print()
    tree = DecisionTree(5)
    tree.fit(X,y)
    y_hat_tree_train = tree.predict(X)
    error_tree_train = 1- np.mean(y_hat_tree_train == y)
    y_hat_tree_test = tree.predict(X_test)
    error_tree_test = 1- np.mean(y_hat_tree_test == y_test)
    print(f"The training error for tree with maximum depth of 5 is {np.round(error_tree_train,4)*100:.1f}%")
    print(f"The testing error for tree with maximum depth of 5 is {np.round(error_tree_test,4)*100:.1f}%")

    
    knn = KNN(1)
    knn.fit(X,y)
    plot_classifier(knn,X,y)
    fname = Path("..", "figs", "q1_knn_1.pdf")
    plt.savefig(fname)
    plt.close() 

    knn_sklearn = KNeighborsClassifier(1)
    knn_sklearn.fit(X, y)
    plot_classifier(knn_sklearn, X, y)
    fname = Path("..", "figs", "q1_knn_1_sklearn.pdf")
    plt.savefig(fname)
    plt.close() 

    #raise NotImplementedError()



@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    n,d = X.shape
    ks = list(range(1, 30, 4))
    fold_n = 10
    fold_size = n//fold_n
    accuracy = np.zeros((fold_n, len(ks)))
    test_accs = np.zeros(len(ks))
    train_accs = np.zeros(len(ks))
    
    for k_idx in range(len(ks)):
        for fold_idx in range(10):
            mask = np.ones(n, dtype=bool)   # reset mask
            mask [fold_idx*fold_size:(fold_idx+1)*fold_size] = False
            X_fold_train = X[mask,:] 
            y_fold_train = y[mask]
            X_fold_test = X[~mask,:]
            y_fold_test = y[~mask]
            knn = KNN(ks[k_idx])
            knn.fit(X_fold_train,y_fold_train)
            y_fold_hat = knn.predict(X_fold_test)
            accuracy[fold_idx, k_idx] = np.mean(y_fold_hat == y_fold_test)
        knn = KNN(ks[k_idx])
        knn.fit(X,y)
        y_hat = knn.predict(X_test)
        y_hat_train = knn.predict(X)
        test_accs[k_idx] = np.mean(y_hat == y_test)
        train_accs[k_idx] = np.mean(y_hat_train == y)
            ## accuracy is a matrix with columns be each k (for KNN) and the rows are each fold (10-fold)
    cv_accs =  np.mean(accuracy, axis = 0)
    plt.plot(ks, cv_accs, linestyle = '-', color = 'red', label='Cross-Validation Accuracy')
    plt.plot(ks, test_accs, linestyle = '-', color = 'blue', label='Test Accuracy')
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("k-NN Performance")
    plt.legend()
    fname = Path("..", "figs", "q2_test_csv_comp.pdf")
    plt.savefig(fname)
    plt.close() 

    plt.plot(ks, train_accs, linestyle = '-', color = 'purple', label='Train Accuracy')
    plt.xlabel("Value of k")
    plt.ylabel("Training Accuracy")
    plt.title("k-NN Performance")
    plt.legend()
    fname = Path("..", "figs", "q2_train_error.pdf")
    plt.savefig(fname)
    plt.close() 

    #raise NotImplementedError()



@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    print(wordlist[72,])
    print(wordlist[X[802,:]])
    print(groupnames[y[802]])



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model_w_lap = NaiveBayesLaplace(num_classes=4)
    model_w_lap.fit(X, y)
    model_wo_lap = NaiveBayes(num_classes=4)
    model_wo_lap.fit(X, y)

    diff_rel = (model_w_lap.p_xy - model_wo_lap.p_xy)/model_w_lap.p_xy * 100
    minimum, q1, q2, q3, maximum = np.percentile(diff_rel, [0, 25, 50, 75, 100])


    print()
    print("difference between p_xy with and without Laplace")
    print(f"Minimum: {minimum: .2f}%")
    print(f"Q1 (25th percentile): {q1: .2f}%")
    print(f"Q2 (Median): {q2: .2f}%")
    print(f"Q3 (75th percentile): {q3:.2f}%")
    print(f"Maximum: {maximum: .2f}%") 
    print()


    model_w_lap_large_beta = NaiveBayesLaplace(num_classes=4, beta=10000)
    model_w_lap_large_beta.fit(X, y)  
    
    p_xy_large_beta = model_w_lap_large_beta.p_xy * 100
    minimum, q1, q2, q3, maximum = np.percentile(p_xy_large_beta, [0, 25, 50, 75, 100])

    print("p_xy with beta = 10000")
    print(f"Minimum: {minimum: .2f}%")
    print(f"Q1 (25th percentile): {q1: .2f}%")
    print(f"Q2 (Median): {q2: .2f}%")
    print(f"Q3 (75th percentile): {q3:.2f}%")
    print(f"Maximum: {maximum: .2f}%") 



    #raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
    print()
    print("Random tree")
    evaluate_model(RandomTree(max_depth=np.inf))
    print()
    print("Random forest")
    evaluate_model(RandomForest(max_depth=np.inf, num_trees= 50))


    #raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "q5_kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]
    model = Kmeans(k=4)
    model.fit(X)
    # Local Minimum
    # Stop when no point changes assignment
    #raise NotImplementedError()

    best_y = model.predict(X)
    best_error = model.error(X, best_y, model.means)

    for _ in range(49):
        model.fit(X)
        current_y = model.predict(X)
        current_error = model.error(X, current_y, model.means)
        if (current_error < best_error):
                best_y = current_y
                best_error = current_error

    print(best_error)
    plt.scatter(X[:, 0], X[:, 1], c=best_y, cmap="jet")

    fname = Path("..", "figs", "q5_kmeans_lowest_error.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")
    plt.close()
    #raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]
    k = range(1,11)
    best_error = np.zeros(10)
    for k_indx in range(0,10):
        model = Kmeans(k=k[k_indx])
        model.fit(X)
        best_y = model.predict(X)
        best_error[k_indx] = model.error(X, best_y, model.means)

        for _ in range(49):
            model.fit(X)
            current_y = model.predict(X)
            current_error = model.error(X, current_y, model.means)
            if (current_error < best_error[k_indx]):
                    best_y = current_y
                    best_error[k_indx] = current_error

    plt.plot(k, best_error, linestyle = '-', color = 'red', label='Error vs k in k-clustering')
    fname = Path("..", "figs", "q5_error_vs_k_clustering.png")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.title("Error vs k in k-clustering")
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    main()
