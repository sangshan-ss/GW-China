import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# calculate precipitation formed by green water from each source province to each sink province and sort
def data_cal_sort():
    data_pre_relative = pd.read_csv('../data/prec_source_zonal0601.csv')
    pro_ETpre_relative = np.array(data_pre_relative.iloc[0:31,1:32])
    data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
    pro_name_list = data_h['province'].tolist()
    # unit km3
    pro_pre_km3_total = np.array(data_h['average_pre(108m3)']).reshape((1,31))/10
    # unit mm
    pro_pre_mm_total = np.array(data_h['average_pre(mm)']).reshape((1,31))
    # pre formed ET (mm)
    pro_ETpre_mm = pro_ETpre_relative * pro_pre_mm_total
    
    # pre formed ET (km3)
    pro_ETpre_km3 = pro_ETpre_relative * pro_pre_km3_total

    # domestic pre by ET
    pro_ETpre_km3_sum  =pro_ETpre_km3.sum(axis=1)
    # pre from domestic ET
    pro_ETpre_km3_dom = pro_ETpre_km3.sum(axis=0)

    # sort
    sorted_index = np.argsort(pro_ETpre_km3_sum)[::-1]
    # pre from domestic ET
    pro_ETpre_km3_dom_sort = pro_ETpre_km3_dom[sorted_index]
    # domestic pre by ET
    pro_ETpre_km3_sum_sort = pro_ETpre_km3_sum[sorted_index]
    # pre formed ET (mm)
    pro_ETpre_mm_sort = pro_ETpre_mm[sorted_index, :]
    pro_ETpre_mm_sort = pro_ETpre_mm_sort[:,sorted_index]
    
    pro_name_list_sort = np.array(pro_name_list)[sorted_index]

    return pro_ETpre_mm_sort,pro_ETpre_km3_dom_sort,pro_ETpre_km3_sum_sort,pro_name_list_sort

pro_ETpre_mm_sort,pro_ETpre_km3_dom_sort,pro_ETpre_km3_sum_sort,pro_name_list_sort = data_cal_sort()

# heatmap plot
def heatmap(ax):
    for i in range(pro_ETpre_mm_sort.shape[0]):
        for j in range(pro_ETpre_mm_sort.shape[1]):
            text_ET = round(pro_ETpre_mm_sort[i,j],1)
            formatted_text = f"{text_ET:.1f}".lstrip('0')
            if pro_ETpre_mm_sort[i, j] > 70:
                ax.text(j,i,int(round(pro_ETpre_mm_sort[i,j],0)), ha='center', va='center', color='white',fontsize=6)
            elif 0 < pro_ETpre_mm_sort[i, j] < 1:
                ax.text(j,i,formatted_text, ha='center', va='center', color='darkred',fontsize=6)
            else:
                ax.text(j,i,int(round(pro_ETpre_mm_sort[i,j],0)), ha='center', va='center', color='darkred',fontsize=6)

    ax.set_xticks(np.arange(len(pro_name_list_sort)), labels=pro_name_list_sort, rotation=90, rotation_mode='anchor', ha='right',va='center', color='green')
    ax.set_xlabel('Sink province', color='green',weight='bold',fontsize=12)
    ax.set_ylabel('Source province',color='#2E8B57',weight='bold',fontsize=12)
    ax.set_yticks(np.arange(len(pro_name_list_sort)), labels=pro_name_list_sort,color = '#2E8B57')
    
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='both', pad=1)

    bins = [0,1,10,30,50,70,100]
    nbin = len(bins)-1
    cmap = cm.get_cmap('Greens', nbin)
    norm = mcolors.BoundaryNorm(bins, nbin)

    heatmap = ax.imshow(pro_ETpre_mm_sort, cmap=cmap, norm=norm, origin='lower')

    cbar = fig.colorbar(heatmap, ax=ax, location='bottom', pad=0.15, extend='max')
    cbar.set_ticks(bins)
    cbar.set_ticklabels(['0','1','10','30','50','70','100'])
    cbar.ax.set_position([ax.get_position().xmin+ax.get_position().width/2-0.2, 0.16, 0.4, 0.02])
    cbar.set_label('Precipitation formed by green water from source to sink provinces (mm)',fontsize=12)

def barh_right(axx):
    axx.barh(np.arange(len(pro_name_list_sort)), pro_ETpre_km3_sum_sort, height=0.8,color='#2E8B57')
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)
    axx.xaxis.tick_bottom()
    x_ticks2 = [200,400]
    axx.set_xticks(x_ticks2)
    axx.margins(y=0.005)
    axx.set_yticklabels([])
    axx.get_yaxis().set_visible(False)
    axx.set_title('Green water formed precipitation in\ndomestic sink provinces',rotation = 270, y=0.1, x=1)
    axx.text(480,-1.8,'km\u00B3',color='black',fontsize=12)
    for i in range(len(pro_name_list_sort)):
        axx.text(pro_ETpre_km3_sum_sort[i]+30,i,int(round(pro_ETpre_km3_sum_sort[i],0)),ha='right', va='center',rotation = 270, color='black',fontsize=6)

def bar_top(axx):
    axx.bar(np.arange(len(pro_name_list_sort)), pro_ETpre_km3_dom_sort, color='green',width=0.75)
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)
    axx.xaxis.tick_bottom()
    axx.margins(x=0.005)
    axx.set_xticklabels([])
    axx.get_xaxis().set_visible(False)
    axx.yaxis.tick_left()
    y_ticks = [200,400]
    axx.set_yticks(y_ticks)
    axx.set_title('Precipitation formed by green water from\ndomestic source provinces',y=0.75, x=0.5)
    axx.text(-3,450,'km\u00B3',color='black',fontsize=12)
    for i in range(len(pro_name_list_sort)):
        axx.text(i, pro_ETpre_km3_dom_sort[i]+20,int(round(pro_ETpre_km3_dom_sort[i],0)),ha='center', va='center', color='black',fontsize=6)

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111) 
heatmap(ax)
ax2 = fig.add_axes([ax.get_position().xmin+ax.get_position().width,ax.get_position().ymin,ax.get_position().width/5,ax.get_position().height])
barh_right(ax2)
ax3 = fig.add_axes([ax.get_position().xmin,ax.get_position().ymax,4*ax.get_position().height/5,ax.get_position().width/5])
bar_top(ax3)

fig.savefig('../figure/figure_2.svg', dpi=300)