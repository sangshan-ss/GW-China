import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
agr_GDP = np.array(data_h['agr_GDP_2020RMB(108)'])/10
food_pro = np.array(data_h['food_production(wan_t)'])/100
total_GDP = np.array(data_h['sum_GDP_2020RMB(108)'])/10
population = np.array(data_h['population(wanren)'])/100

def scatter_linregress(ax,x,y,color1,color2,xlabel,ylabel,label,unit1,unit2):
    # fit line
    slope, intercept, r, _, _ = linregress(x, y)
    fit_line = slope * x + intercept
    
    ax.plot(x, fit_line, color=color2)
    ax.scatter(x, y,color = color1)
    
    ax.set_xlabel(xlabel,fontsize = 14)
    ax.set_ylabel(ylabel,fontsize = 14)
    ax.tick_params(axis='both', labelsize=14)

    for i in ['top', 'right']:
        ax.spines[i].set_visible(False)
    # annotation
    ax.text(0.15,0.95,f'R: {round(r,2)}',transform=ax.transAxes, fontdict={'size': '16', 'color': 'darkred'},
                horizontalalignment='center', verticalalignment='center')

    # num
    ax.text(-0.12,1.1,label,transform=ax.transAxes, fontdict={'size': '16', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    # unit
    ax.text(0,1.03,unit1,transform=ax.transAxes, fontdict={'size': '16', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    ax.text(0.98,-0.09,unit2,transform=ax.transAxes, fontdict={'size': '16', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')


fig = plt.figure(figsize=(12, 6),dpi=300)
plt.subplots_adjust(wspace=0.3)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
scatter_linregress(ax1,food_pro,agr_GDP,'royalblue','darkorange','Food production','Agriculture GDP',
                   '(a)','billion RMB','million ton')
scatter_linregress(ax2,population,total_GDP,'forestgreen','mediumorchid','Population','Total GDP',
                   '(b)','billion RMB','million pop')

fig.savefig('../figure/figure_A6.svg', dpi = 300)