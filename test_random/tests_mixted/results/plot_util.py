#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import pandas
import seaborn as sns
#get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
from skmultiflow.visualization.evaluation_visualizer import EvaluationVisualizer
csv_list = os.listdir()
plt.rcParams.update({'figure.max_open_warning': 0})
csv_list.remove("plot_util.ipynb")
#csv_list.remove(".ipynb_checkpoints")
csv_list.remove("plot_util.py")

# In[ ]:





# In[10]:


def plot(filename):
    df = pandas.read_csv(filename,skiprows=6)
    plt.figure(figsize=(20,10))
    df = df.rename(index=str, columns={"id": "Max_iter"})
    if("mean_kappa_[HT]" in df.columns):
        fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20, 10))
        fig.suptitle(filename)

        df.plot(x="Max_iter", y=["mean_acc_[HT]","mean_acc_[HATT]"],ax=axes[0,0])
        df.plot(x="Max_iter", y=["current_acc_[HT]","current_acc_[HATT]"],ax=axes[1,0])
        
        df.plot(x="Max_iter", y=["mean_kappa_[HT]","mean_kappa_[HATT]"],ax=axes[0,1])
        df.plot(x="Max_iter", y=["current_kappa_[HATT]","current_kappa_[HT]"],ax=axes[1,1])
        plt.savefig("../results_png/" + filename[:-3]+"png")
    else:
        print(filename)


# In[11]:


for file in csv_list:
    plot(file)


# In[ ]:




