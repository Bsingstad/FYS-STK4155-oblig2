import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


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