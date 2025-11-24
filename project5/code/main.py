#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(lr_model, X_train, y_train)
    savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # kernel logistic regression with a polynomial kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel_pol = PolynomialKernel(2)
    klr_model_pol = KernelClassifier(loss_fn, optimizer, kernel_pol)
    klr_model_pol.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model_pol.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model_pol.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model_pol, X_train, y_train)
    savefig("logRegPol.png", fig)

    # kernel logistic regression with a RBF kernel
    kernel_rbf = GaussianRBFKernel(0.5)
    klr_model_rbf = KernelClassifier(loss_fn, optimizer, kernel_rbf)
    klr_model_rbf.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model_rbf.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model_rbf.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model_rbf, X_train, y_train)
    savefig("logRegRBF.png", fig)


@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])
    
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val
    optimizer = GradientDescentLineSearch()
    for i in range(len(sigmas)):
        for j in range(len(lammys)):
            loss_fn = KernelLogisticRegressionLossL2(lammys[j])
            kernel_rbf = GaussianRBFKernel(sigmas[i])
            klr_model_rbf = KernelClassifier(loss_fn, optimizer, kernel_rbf)
            klr_model_rbf.fit(X_train, y_train)
            train_errs[i, j] = np.mean(klr_model_rbf.predict(X_train) != y_train)
            val_errs[i, j] = np.mean(klr_model_rbf.predict(X_val) != y_val)


    # Hyperparameters for minimum lambda and sigma
    ## Training Error
    min_train_err_inx = np.unravel_index(np.argmin(train_errs, axis=None), train_errs.shape)
    min_train_error_sigma = sigmas[min_train_err_inx[0]]
    min_train_error_lammy = lammys[min_train_err_inx[1]]
    min_train_error = train_errs[min_train_err_inx]

    print(f"Sigma for minimum training error is {min_train_error_sigma:.4f}")
    print(f"Lambda for minimum training error is {min_train_error_lammy:.4f}")
    print(f"Minimum training error is {min_train_error:.1%}")

    loss_fn = KernelLogisticRegressionLossL2(min_train_error_lammy)
    kernel_rbf = GaussianRBFKernel(min_train_error_sigma)
    klr_model_rbf_min_training_error = KernelClassifier(loss_fn, optimizer, kernel_rbf)
    klr_model_rbf_min_training_error.fit(X_train, y_train)
    fig = plot_classifier(klr_model_rbf_min_training_error, X_train, y_train)
    savefig("logRegRBF_min_train_error.png", fig)

    ## Validation Error
    min_val_err_inx = np.unravel_index(np.argmin(val_errs, axis=None), val_errs.shape)
    min_val_error_sigma = sigmas[min_val_err_inx[0]]
    min_val_error_lammy = lammys[min_val_err_inx[1]]
    min_valid_error = val_errs[min_val_err_inx]

    print(f"Sigma for minimum validation error is {min_val_error_sigma:.4f}")
    print(f"Lambda for minimum validation error is {min_val_error_lammy:.4f}")
    print(f"Minimum validation error is {min_valid_error:.1%}")

    loss_fn = KernelLogisticRegressionLossL2(min_val_error_lammy)
    kernel_rbf = GaussianRBFKernel(min_val_error_sigma)
    klr_model_rbf_min_valid_error = KernelClassifier(loss_fn, optimizer, kernel_rbf)
    klr_model_rbf_min_valid_error.fit(X_train, y_train)
    fig = plot_classifier(klr_model_rbf_min_valid_error, X_train, y_train)
    savefig("logRegRBF_min_valid_error.png", fig)

    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)

@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)

    PCA_encoder = PCAEncoder(2)
    PCA_encoder.fit(X_train_standardized)
    W = PCA_encoder.W
    Z = X_train_standardized @ W.T
    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1])
    for i in random_is:
        xy = Z[i, [0, 1]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_PCA.png", fig)
    plt.close(fig)

    print(f"the trait with largest absolute value of Z1 is {trait_names[np.argmax(np.abs(W), axis = 1)[0]]}")
    print(f"the trait with largest absolute value of Z2 is {trait_names[np.argmax(np.abs(W), axis = 1)[1]]}")

    reconstruction_loss = np.linalg.norm(Z@W-X_train_standardized)**2
    X_train_standardized_norm = np.linalg.norm(X_train_standardized)**2
    print(f"variance explained is {1-reconstruction_loss/X_train_standardized_norm:.2%}")

    PCA_encoder = PCAEncoder(5)
    PCA_encoder.fit(X_train_standardized)
    W = PCA_encoder.W
    Z = X_train_standardized @ W.T
    reconstruction_loss = np.linalg.norm(Z@W-X_train_standardized)**2
    X_train_standardized_norm = np.linalg.norm(X_train_standardized)**2
    print(f"variance explained is {1-reconstruction_loss/X_train_standardized_norm:.2%}")

    



@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    loss = LeastSquaresLoss()
    learning_rate = ConstantLR(0.0003)
    for i in [1,10,100]:
        optimizer = StochasticGradient (learning_rate_getter=learning_rate, 
                                    batch_size=i, 
                                    base_optimizer=GradientDescent(),
                                    max_evals= 10)
        model  = LinearModel(loss, optimizer)
        model.fit(X_train, y_train)
        train_error = np.mean((model.predict(X_train) - y_train)**2)
        valid_error = np.mean((model.predict(X_val) - y_val)**2)
        print(f"Training error (MSE) for batch size of {i} is {train_error:.3f}")
        print(f"Validation error (MSE) for batch size of {i} is {valid_error:.3f}")


@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    loss = LeastSquaresLoss()
    c= 0.1
    learning_rate_CLR = ConstantLR(c)
    learning_rate_ILR = InverseLR(c)
    learning_rate_ISLR = InverseSquaredLR(c)
    learning_rate_ISqrtLR = InverseSqrtLR(c)

    for (learning_rate, names) in [(learning_rate_CLR,"CLR"), 
                                   (learning_rate_ILR, "ILR"), 
                                   (learning_rate_ISLR, "ISLR"), 
                                   (learning_rate_ISqrtLR, "ISqrtLR")]:
        optimizer = StochasticGradient (learning_rate_getter=learning_rate, 
                                    batch_size=10, 
                                    base_optimizer=GradientDescent(),
                                    max_evals= 50)
        model  = LinearModel(loss, optimizer)
        model.fit(X_train, y_train)
        fig, ax = plt.subplots()
        ax.plot(model.fs, marker="o")
        ax.set_xlabel("Gradient descent iterations")
        ax.set_ylabel("Objective function f value")
        ax.set_title(f"learning rate is {names}")
        dynamic_filename = f'gd_curve_iteration_{names}.png'
        savefig(dynamic_filename, fig)




if __name__ == "__main__":
    main()
