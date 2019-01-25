#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def hist(np_var, title, fig_num=1):
    plt.figure(fig_num)
    plt.hist(np_var, bins='auto')
    plt.title(title)
    plt.show()
    return None

def plot_3D(X_list, Y_list, Z_list, title, 
            X_label, Y_label, Z_label, fig_num=1, 
            label_list=[''], color_style_list=[['b','-']], 
            is_showing_start_and_goal=False,
            N_data_display=-1,
            is_auto_line_coloring_and_styling=False):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(Z_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_color_list = ['b','g','r','c','m','y','k']
        all_style_list = ['-','--','-.',':']
        color_style_list = [[c,s] for c in all_color_list for s in all_style_list]
    assert (N_dataset <= len(color_style_list))
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')
    for d in range(N_dataset):
        X = X_list[d][:N_data_display]
        Y = Y_list[d][:N_data_display]
        Z = Z_list[d][:N_data_display]
        color = color_style_list[d][0]
        linestyle = color_style_list[d][1]
        labl = label_list[d]
        ax.plot(X, Y, Z, c=color, ls=linestyle, label=labl)
        if (is_showing_start_and_goal):
            ax.scatter(X_list[d][0], Y_list[d][0], Z_list[d][0], 
                       c=color, label='start '+labl, marker='o')
            ax.scatter(X_list[d][-1], Y_list[d][-1], Z_list[d][-1], 
                       c=color, label='end '+labl, marker='^', s=200)
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_zlabel(Z_label)
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    plt.show()
    return None

def plot_2D(X_list, Y_list, title, 
            X_label, Y_label, fig_num=1, 
            label_list=[''], color_style_list=[['b','-']], 
            is_showing_start_and_goal=False,
            N_data_display=-1,
            is_auto_line_coloring_and_styling=False):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_color_list = ['b','g','r','c','m','y','k']
        all_style_list = ['-','--','-.',':']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_dataset <= len(color_style_list))
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111)
    for d in range(N_dataset):
        X = X_list[d][:N_data_display]
        Y = Y_list[d][:N_data_display]
        color = color_style_list[d][0]
        linestyle = color_style_list[d][1]
        labl = label_list[d]
        ax.plot(X, Y, c=color, ls=linestyle, label=labl)
        if (is_showing_start_and_goal):
            ax.scatter(X_list[d][0], Y_list[d][0], 
                       c=color, label='start '+labl, marker='o')
            ax.scatter(X_list[d][-1], Y_list[d][-1], 
                       c=color, label='end '+labl, marker='^', s=200)
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    plt.show()
    return None

def scatter_3D(X_list, Y_list, Z_list, title, 
               X_label, Y_label, Z_label, fig_num=1, 
               label_list=[''], color_style_list=[['b','o']],
               is_auto_line_coloring_and_styling=False):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(Z_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_color_list = ['b','g','r','c','m','y','k']
        all_style_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_dataset <= len(color_style_list)), "Not enough color style variations to cover all datasets!"
    else:
        assert (N_dataset <= len(color_style_list))
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')
    for d in range(N_dataset):
        X = X_list[d]
        Y = Y_list[d]
        Z = Z_list[d]
        color = color_style_list[d][0]
        markerstyle = color_style_list[d][1]
        labl = label_list[d]
        ax.scatter(X, Y, Z, c=color, marker=markerstyle, label=labl)
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_zlabel(Z_label)
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
    plt.show()
    return None

def scatter_2D(X_list, Y_list, title, 
               X_label, Y_label, fig_num=1, 
               label_list=[''], color_style_list=[['b','o']],
               is_auto_line_coloring_and_styling=False):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_color_list = ['b','g','r','c','m','y','k']
        all_style_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_dataset <= len(color_style_list)), "Not enough color style variations to cover all datasets!"
    else:
        assert (N_dataset == len(color_style_list))
    plt.figure(fig_num)
    for d in range(N_dataset):
        X = X_list[d]
        Y = Y_list[d]
        color = color_style_list[d][0]
        markerstyle = color_style_list[d][1]
        labl = label_list[d]
        plt.scatter(X, Y, c=color, marker=markerstyle, label=labl)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def subplot_ND(NDtraj_list, title, 
               Y_label_list, fig_num=1, 
               label_list=[''], color_style_list=[['b','o']],
               is_auto_line_coloring_and_styling=False):
    assert (len(NDtraj_list) >= 1)
    N_traj_to_plot = len(NDtraj_list)
    assert (len(label_list) == N_traj_to_plot)
    D = NDtraj_list[0].shape[1]
    assert (len(Y_label_list) == D)
    if (is_auto_line_coloring_and_styling):
        all_color_list = ['b','g','r','c','m','y','k']
        all_style_list = ['-','--','-.',':']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_traj_to_plot <= len(color_style_list)), "Not enough color style variations to cover all datasets!"
    
    ax = [None] * D
    fig, ax = plt.subplots(D, sharex=True, sharey=True)
    
    for n_traj_to_plot in range(N_traj_to_plot):
        assert (NDtraj_list[n_traj_to_plot].shape[1] == D)
        traj_label = label_list[n_traj_to_plot]
        for d in range(D):
            if (n_traj_to_plot == 0):
                if (d == 0):
                    ax[d].set_title(title)
                ax[d].set_ylabel(Y_label_list[d])
                if (d == D-1):
                    ax[d].set_xlabel('Time Index')
            traj = NDtraj_list[n_traj_to_plot][:,d]
            color = color_style_list[n_traj_to_plot][0]
            linestyle = color_style_list[n_traj_to_plot][1]
            ax[d].plot(traj, 
                       c=color, ls=linestyle, 
                       label=traj_label)
    ax[0].legend()
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
#    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    return None


if __name__ == '__main__':
    plt.close('all')
    
    N_data_points = 1000
    x = np.array(range(N_data_points))/(1.0 * (N_data_points-1))
    
    x_pow_1 = np.power(x,1)
    x_pow_8 = np.power(x,8)
    D_display = 2
    data_dim_list = [None] * D_display
    for i in range(D_display):
        data_dim_list[i] = list()
        if (i == 0):
            for j in range(2):
                data_dim_list[i].append(x)
        elif (i == 1):
            data_dim_list[i].append(x_pow_1)
            data_dim_list[i].append(x_pow_8)
    plot_2D(X_list=data_dim_list[0], 
            Y_list=data_dim_list[1], 
            title='Powers of x', 
            X_label='x', 
            Y_label='Value', 
            fig_num=2, 
            label_list=['x','x^8'], 
            color_style_list=[['b','-'],['r','-']])
    
    x_pow_2 = np.power(x,2)
    x_pow_3 = np.power(x,3)
    x_pow_4 = np.power(x,4)
    x_pow_5 = np.power(x,5)
    TwoDtraj_list = [np.zeros((N_data_points, 2))] * 2
    TwoDtraj_list[0] = np.vstack([x_pow_1, x_pow_8, x_pow_4]).T
    TwoDtraj_list[1] = np.vstack([x_pow_2, x_pow_3, x_pow_5]).T
    subplot_ND(NDtraj_list=TwoDtraj_list, 
               title='SubPlot Test', 
               Y_label_list=['x','y','z'], 
               fig_num=3, 
               label_list=['pow 1 vs pow 8 vs pow 4', 'pow 2 vs pow 3 vs pow 5'], 
               is_auto_line_coloring_and_styling=True)