import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_frankesfunction(x,y,z,poly):
    fig = plt.figure(figsize=(32,12))
    ax = fig.gca(projection ='3d')
    surf = ax.plot_surface(x,y,z.reshape(x.shape),cmap=cm.coolwarm, linewidth = 0, antialiased=False)
    ax.set_zlim(-0.10,1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf,shrink=0.5, aspect=5)
    fig.suptitle("A {} degree polynomial fit of Franke function using OLS".format(poly) ,fontsize="40", color = "black")
    fig.show()

def plot_frankesfunction_ridge(x,y,z,poly,lambda_):
    fig = plt.figure(figsize=(32,12))
    ax = fig.gca(projection ='3d')
    surf = ax.plot_surface(x,y,z.reshape(x.shape),cmap=cm.coolwarm, linewidth = 0, antialiased=False)
    ax.set_zlim(-0.10,1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf,shrink=0.5, aspect=5)
    fig.suptitle("A {} degree polynomial fit of Franke's function using Ridge with lamba {}".format(poly,lambda_) ,fontsize="40", color = "black")
    fig.show()

def compute_modified_confusion_matrix_nonorm(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        #####normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0#/normalization

    return A

def plot_classes(classes,y_ohe,plot_name = "myplot"):
    plt.style.use('seaborn-paper')
    plt.figure(figsize=(20,16))
    plt.bar(x=classes,height=y_ohe.sum(axis=0))
    plt.title("Distribution of Diagnosis", color = "black")
    plt.tick_params(axis="both", colors = "black")
    plt.xlabel("Diagnosis", color = "black")
    plt.ylabel("Count", color = "black")
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize = 20)
    plt.savefig(plot_name + ".png")
    plt.show()
