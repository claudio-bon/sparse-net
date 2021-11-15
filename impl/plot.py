# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
import numpy as np
from IPython.display import display
import matplotlib.ticker as plticker

sns.set_style("whitegrid")

DEBUGGING = False

data_folder = "test_results"
measures_folder = path.join(data_folder, "measures")
plot_folder = path.join(data_folder, "plots")

openmp_folder = "openmp"
openmp_measures_folder = path.join(openmp_folder, measures_folder)
openmp_plots_folder = path.join(openmp_folder, plot_folder)

cuda_folder = "cuda"
cuda_measures_folder = path.join(cuda_folder, measures_folder)
cuda_plots_folder = path.join(cuda_folder, plot_folder)


def plot_figure(x, y, data=None, hue=None, plot_title=None, xlabel=None, ylabel=None, filename=None,
                legend_title=None, legend_labels=None, fig_size=(5.0,3.5), xticks_rotation=None):
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue)
    
    #Set plot title, x axis label and y axis label
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=14)
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=12)
    
    sns.despine()
    
    #Setup legend
    if legend_title is not None and legend_labels is not None:
        plt.legend(labels=legend_labels, title = legend_title)
        plt.setp(ax.get_legend().get_texts(), fontsize=11) # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize=12) # for legend title
    #Rotate ticks on the x axis
    if xticks_rotation is not None:
        plt.xticks(rotation=xticks_rotation)
    #Save plot
    if filename is not None:
        plt.savefig(filename)
    plt.show()




#--------------------------------- OPENMP ---------------------------------#

#------------------------------ ELAPSED TIME ------------------------------#


df_elapsed = pd.read_csv(path.join(openmp_measures_folder, "elapsed_time.csv"))

if DEBUGGING:
    display(df_elapsed.head(12))
    print()

df_elapsed = df_elapsed.loc[df_elapsed["pt"]!="FULL"]
df_elapsed["pt"] = df_elapsed["pt"].apply(lambda x: x.capitalize())


if DEBUGGING:
    display(df_elapsed.head(12))
    print()



plot_figure("p", "t",
            hue="pt",
            data=df_elapsed,
            plot_title="Elapsed Time",
            ylabel="sec",
            xlabel="Number of threads",
            filename=path.join(openmp_plots_folder,"openmp_elapsed-time.png"),
            legend_title="Parallelism Type",
            legend_labels=["Outer","Inner","Sequential"])



#--------------------------------- SPEEDUP ---------------------------------#


df_strong = pd.read_csv(path.join(openmp_measures_folder, "strong_scale.csv"))
df_strong.drop(columns="pt", inplace=True)
df_strong = df_strong.groupby(["K","p"]).mean().reset_index()

if DEBUGGING:
    print()
    print(df_strong["K"].unique())
    print()
    print("\ndf_outer:\n")
    display(df_strong)
    print()

num_layers_list = np.sort(df_strong["K"].unique())
T_0 = {k: df_strong.loc[df_strong["K"]==k]["t"].values[0] for k in num_layers_list}
speedup = [T_0[k]/t for k in np.sort(df_strong["K"].unique())
                    for t in df_strong.loc[df_strong["K"]==k]["t"].values]

df_strong["speedup"] = speedup

if DEBUGGING:
    print("\ndf_outer:\n")
    display(df_strong)
    print()


plot_figure("p", "speedup",
            hue="K",
            data=df_strong,
            plot_title="Speedup",
            ylabel="Speedup",
            xlabel="Number of threads",
            filename=path.join(openmp_plots_folder,"openmp_speedup_outer.png"),
            legend_title="Number of Layers",
            legend_labels=df_strong["K"].unique())




#------------------------------ STRONG SCALING ------------------------------#

if DEBUGGING:
    print("\nK=250:\n")
    display(df_strong.loc[df_strong["K"]==250][["p","speedup"]].values)
    print()

strong_sc = [s/p for k in num_layers_list
                 for p,s in df_strong.loc[df_strong["K"]==k][["p","speedup"]].values]

df_strong["strong_sc"] = strong_sc


plot_figure("p", "strong_sc",
            hue="K",
            data=df_strong,
            plot_title="Strong Scaling Efficiency",
            ylabel="Strong Scaling Efficiency",
            xlabel="Number of threads",
            filename=path.join(openmp_plots_folder,"openmp_strong-scaling-efficiency_outer.png"),
            legend_title="Number of Layers",
            legend_labels=df_strong["K"].unique())



#------------------------------- WEAK SCALING -------------------------------#


df_weak = pd.read_csv(path.join(openmp_measures_folder, "weak_scale.csv"))
df_weak.drop(columns="pt", inplace=True)
df_weak = df_weak.groupby(["K","p"]).mean().reset_index()

if DEBUGGING:
    print("\ndf_weak:\n")
    display(df_weak)
    print()

num_layers_list = np.sort(df_weak["K"].unique())

T_0 = {k: df_weak.loc[df_weak["K"]==k]["t"].values[0] for k in num_layers_list}

weak_sc = [T_0[k]/t for k in num_layers_list
                    for t in df_weak.loc[df_weak["K"]==k]["t"].values]
df_weak["weak_sc"] = weak_sc


plot_figure("p", "weak_sc",
            hue="K",
            data=df_weak,
            plot_title="Weak Scaling Efficiency",
            ylabel="Weak Scaling Efficiency",
            xlabel="Number of threads",
            filename=path.join(openmp_plots_folder,"openmp_weak-scaling-efficiency_outer.png"),
            legend_title="Number of Layers",
            legend_labels=df_weak["K"].unique())




#------------------------------------ CUDA ------------------------------------#


#---------------------------- SPEEDUP GPU VS. CPU -----------------------------#


df_seq = pd.read_csv(path.join(openmp_measures_folder, "measures_openmp.csv"))
df_seq.drop(columns="pt", inplace=True)
df_seq = df_seq.groupby(["K","N"]).mean().reset_index()

df_cuda = pd.read_csv(path.join(cuda_measures_folder, "measures_cuda.csv"))
df_cuda = df_cuda.groupby(["K","N","nops"]).mean().reset_index()


if DEBUGGING:
    print("\nSequential:")
    display(df_seq)
    print()
    
    print("\nCUDA:")
    display(df_cuda)
    print()


num_layers_list = np.sort(df_cuda["K"].unique())
speedup_gpu_vs_cpu = [c/g for k in num_layers_list
                          for c,g in zip(df_seq.loc[df_seq["K"]==k]["t"].values,
                                         df_cuda.loc[df_cuda["K"]==k]["tot"].values)]
                          
df_cuda["speedup_cpu_vs_gpu"] = speedup_gpu_vs_cpu


if DEBUGGING:
    print("\nCUDA:")
    display(df_cuda)
    print()


plot_figure("N", "speedup_cpu_vs_gpu",
            hue="K",
            data=df_cuda,
            plot_title="Speedup GPU vs. CPU",
            ylabel="Speedup",
            xlabel="N",
            filename=path.join(cuda_plots_folder, "cuda_speedup-gpu-vs-cpu.png"),
            legend_title="Number of Layers",
            legend_labels=df_cuda["K"].unique(),
            xticks_rotation=70,
            fig_size=(5.0,4.6))



#--------------------------------- THROUGHPUT ---------------------------------#


throughput = [nops/(kt*10**9) for nops,kt in zip(df_cuda["nops"].values,
                                                 df_cuda["kt"].values)]

df_cuda["throughput"] = throughput


plot_figure("N", "throughput",
            hue="K",
            data=df_cuda,
            plot_title="Throughput",
            ylabel="GFLOP/s",
            xlabel="N",
            filename=path.join(cuda_plots_folder, "cuda_throughput.png"),
            legend_title="Number of Layers",
            legend_labels=df_cuda["K"].unique(),
            xticks_rotation=70,
            fig_size=(5.0,4.6))