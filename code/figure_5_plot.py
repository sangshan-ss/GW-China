import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# calculate green water's socio-economic value of each source province
def arrow_data(data,unit_data,m1,m2):   
    data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
    data_preelative = pd.read_csv('../data/prec_source_zonal0601.csv')

    pro_ETpreelative = np.array(data_preelative.iloc[0:31,1:32])
    water_withdrawal = np.array(data_h['sum_water_withdrawal(108m3)']).reshape((1,31))
    socioeconomic_data = np.array(data_h[data]).reshape((1,31))
    per_unit_water_socio = np.array(data_h[unit_data]).reshape((1,31))
    
    # m1, m2 unit adjust
    # socio-economic value of each source province's green water when it flows to sink provinces
    water_ECOvalue_outerunit = pro_ETpreelative * socioeconomic_data * m1
    # socio-economic value of each province's green water when it is consumed in source provinces
    water_ECOvalue_innerunit = pro_ETpreelative * water_withdrawal * per_unit_water_socio.reshape((31,1)) * m2

    water_ECOvalue_outerunit_sum = water_ECOvalue_outerunit.sum(axis=1)
    water_ECOvalue_innerunit_sum = water_ECOvalue_innerunit.sum(axis=1)
    water_ECOvalue_outerunit_total= water_ECOvalue_outerunit_sum.sum()
    water_ECOvalue_innerunit_total = water_ECOvalue_innerunit_sum.sum()
    # net socio-economic value of green water from source to sink provinces
    net = water_ECOvalue_outerunit_sum-water_ECOvalue_innerunit_sum
    count = np.sum(net > 0)

    return water_ECOvalue_outerunit_sum,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_total,water_ECOvalue_innerunit_total,count

# scatters refers provicnes in the left and right 
def scatter_axes(ax1,ylim,yticks,ylabel1,ylabel2,unit,num):
    ax1.scatter([0.006]*31,water_ECOvalue_innerunit_sum,color='yellowgreen',alpha=0.8)
    ax1.xaxis.set_visible(False)
    for i in ['top', 'bottom']:
        ax1.spines[i].set_visible(False)
    ax1.set_ylim(0,ylim)
    ax1.set_ylabel(ylabel1,fontsize=26)
    ax1.tick_params(axis='both', labelsize=26)
    ax1.text(-0.13,1.1,unit,transform=ax1.transAxes, fontdict={'size': '26', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    ax1.text(0.5,1.1,num,transform=ax1.transAxes, fontdict={'size': '26', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')

    ax2 = ax1.twinx()
    ax2.set_xlim([0, 1])
    for i in ['top', 'bottom','left']:
        ax2.spines[i].set_visible(False)
    ax2.set_ylim(0,ylim)
    ax2.set_yticks(yticks)
    ax2.tick_params(axis='both', labelsize=26)
    ax2.set_ylabel(ylabel2,rotation = 270, ha='center', va='center',labelpad=8, fontsize=26)

# arrows from left to right plot
def arrow(ax,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_sum,count):
    for d in range(water_ECOvalue_outerunit_sum.shape[0]):
        if water_ECOvalue_outerunit_sum[d] > water_ECOvalue_innerunit_sum[d]:
            color = 'tomato'
        else:
            color = 'royalblue'
        arrow = FancyArrowPatch((0,water_ECOvalue_innerunit_sum[d]), (1.01,water_ECOvalue_outerunit_sum[d]), color=color, arrowstyle='->', 
                                mutation_scale=20, alpha=1)
        ax.add_patch(arrow)

    # legend
    legend1= FancyArrowPatch((0.18,0.99), (0.05,0.99), transform=ax.transAxes, color='tomato', arrowstyle='<-', 
                                mutation_scale=25, alpha=1)
    ax.add_patch(legend1)
    ax.text(0.55, 0.99, f'Increase (num = {count})', transform=ax.transAxes, ha='center', va='center', fontsize=26)

    legend2= FancyArrowPatch((0.18,0.94), (0.05,0.94), transform=ax.transAxes, color='royalblue', arrowstyle='<-', 
                                mutation_scale=25, alpha=1)
    ax.add_patch(legend2)
    ax.text(0.56, 0.94, f'Decrease (num = {31-count})', transform=ax.transAxes, ha='center', va='center', fontsize=26)

# green bar refers net socio-economic value of green water plot
def bar_plot(axx,water_ECOvalue_innerunit_total,water_ECOvalue_outerunit_total,ymin,ymax,unitname):
    axx.bar([1,2.5],[water_ECOvalue_innerunit_total,water_ECOvalue_outerunit_total],color='yellowgreen', width=0.5, alpha = 0.8)
    axx.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=True)
    axx.spines['top'].set_linestyle('dashed')
    axx.spines['bottom'].set_linestyle('dashed')
    axx.spines['left'].set_linestyle('dashed')
    axx.spines['right'].set_linestyle('dashed')
    axx.spines['top'].set_color('white')
    axx.spines['right'].set_color('gray')
    axx.spines['bottom'].set_color('gray')
    axx.spines['left'].set_color('gray')
    axx.set_xticks([1, 2.5])
    axx.set_xticklabels(['Source','Sink'],fontdict={'size': '24', 'color': 'black'})
    axx.set_ylim([ymin,ymax])
    axx.text(0.5,1.15,f'Total net = {int(round(water_ECOvalue_outerunit_total-water_ECOvalue_innerunit_total,0))} {unitname}',
             transform=axx.transAxes,ha='center', va='center', fontsize=24, color='tomato',alpha = 1)

fig = plt.figure(figsize=(25, 10), dpi=300)
plt.subplots_adjust(wspace=0.4)

ax1 = fig.add_subplot(131) 
ax11 = fig.add_axes([0.20,0.65,0.06,0.1])
water_ECOvalue_outerunit_sum,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_total,water_ECOvalue_innerunit_total,count= arrow_data('sum_GDP_2020RMB(108)','per_unit_water_value(rmb_m3)',0.0001,0.0001)

scatter_axes(ax1,3,[0, 0.5, 1, 1.5, 2, 2.5, 3],
             'Source values (trillion CNY)','Sink values (trillion CNY)','(a)','GDP')
bar_plot(ax11,water_ECOvalue_innerunit_total,water_ECOvalue_outerunit_total,25,31.5,'trillion CNY')
arrow(ax1,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_sum,count)

ax2 = fig.add_subplot(132) 
ax21 = fig.add_axes([0.48,0.65,0.06,0.1])
water_ECOvalue_outerunit_sum,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_total,water_ECOvalue_innerunit_total,count = arrow_data('population(wanren)','per_unit_water_pop(pop_m3)',0.01,100)

scatter_axes(ax2,80,[0, 10, 20, 30, 40, 50, 60, 70, 80],
             'Source values (million persons)','Sink values (million persons)','(b)','Population')
bar_plot(ax21,water_ECOvalue_innerunit_total,water_ECOvalue_outerunit_total,605,635,'million persons')
arrow(ax2,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_sum,count)

ax3 = fig.add_subplot(133) 
ax31 = fig.add_axes([0.77,0.65,0.06,0.1])
water_ECOvalue_outerunit_sum,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_total,water_ECOvalue_innerunit_total,count = arrow_data('food_production(wan_t)','per_unit_water_food(ton_m3)',0.01,100)

scatter_axes(ax3,35,[0, 5, 10, 15, 20, 25, 30, 35],
             'Source values (million ton)','Sink values (million ton)','(c)','Food production')
bar_plot(ax31,water_ECOvalue_innerunit_total,water_ECOvalue_outerunit_total,220,310,'million ton')
arrow(ax3,water_ECOvalue_innerunit_sum,water_ECOvalue_outerunit_sum,count)

fig.savefig('../figure/figure_5.svg', dpi = 300)