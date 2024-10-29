import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader

# socio-economic data of each province
def social_economic_data_sort():

    data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
    pro_name_list = data_h['province'].tolist()
    # km3
    pro_wr_km3 = data_h['average_water_resources(108m3)'].values/10
    pro_pre_km3 = data_h['average_pre(108m3)'].values/10
    
    # billion
    pro_GDP = data_h['sum_GDP_2020RMB(108)'].values/10000
    pro_agr = data_h['agr_GDP_2020RMB(108)'].values/10000
    pro_ind = data_h['ind_GDP_2020RMB(108)'].values/10000
    pro_ser = data_h['ser_GDP_2020RMB(108)'].values/10000
    # million
    pro_pop  = data_h['population(wanren)'].values/100
    pro_food = data_h['food_production(wan_t)'].values/100

    # sort
    sorted_index_S = np.argsort(pro_wr_km3)[::-1]
    pro_wr_km3_sort = pro_wr_km3[sorted_index_S]
    pro_name_list_sort_s = np.array(pro_name_list)[sorted_index_S]

    pro_pre_km3_sort = pro_pre_km3[sorted_index_S]
    pro_pop_sort = pro_pop[sorted_index_S]
    pro_food_sort = pro_food[sorted_index_S]

    return (pro_wr_km3,pro_pre_km3,pro_GDP,pro_pop,pro_food,pro_agr,pro_ind,pro_ser,pro_wr_km3_sort,pro_name_list_sort_s,pro_pre_km3_sort,
sorted_index_S,pro_pop_sort,pro_food_sort)

# china province shp
def chinap_code(shp_path,data_value):
    shp_data = gpd.read_file(shp_path)
    pro_info = pd.read_csv('../data/china_province_center_point.csv')
    data_df = pd.concat([pro_info.iloc[:,0],data_value],axis = 1,ignore_index = True)
    data_df = data_df.rename(columns={0: "省代码",1: "datavalue"})
    pic_data = shp_data.merge(data_df, on='省代码')
    return pic_data

# water resource, precipitation, population and food production of green water in source provinces plot
def barh_plot(ax,pro_name_list_sort,data_sort,label,color,legloc,xlabel,xtick):
    ax.barh(np.arange(len(pro_name_list_sort)), data_sort, label=label,height=0.8,color=color)
    ax.xaxis.tick_top()
    ax.margins(y=0.0)
    ax.set_yticks(range(len(pro_name_list_sort)))
    ax.set_yticklabels(pro_name_list_sort_s.tolist())
    ax.invert_yaxis()
    ax.legend(loc=legloc, prop={'size': 16}, frameon=False)
    ax.set_xlabel(xlabel,fontsize = 16)
    ax.xaxis.set_label_coords(0.15, 1.06)
    ax.set_xticks(xtick)
    ax.tick_params(axis='both', labelsize=16)

# china map at each bar top plot
def China_map_plot(ax,chinap_shp,NH_shp,TXA_shp,data_vlaue,bins,colorbar,title):
    axx = fig.add_axes([ax.get_position().x0-0.03, ax.get_position().y1+0.02, ax.get_position().width, 0.2],projection=ccrs.PlateCarree(),frameon=False)
    axx.add_geometries(shpreader.Reader(chinap_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', lw=0.5,alpha=1)
    axx.add_geometries(shpreader.Reader(NH_shp).geometries(),\
                        crs = ccrs.PlateCarree(),facecolor = 'None',lw = 0.5,alpha = 1)
    axx.add_geometries(shpreader.Reader(TXA_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', hatch='\\\\\\\\',lw=0.5,alpha=1) 
    axx.set_extent([70, 140, 2, 50],ccrs.Geodetic())
    axx.set_title(title,fontsize = 16)

    china_data = chinap_code(chinap_shp,pd.DataFrame(data_vlaue))
    nbin = len(bins) - 1
    cmap = plt.cm.get_cmap(colorbar, nbin)
    norm = mcolors.BoundaryNorm(bins, nbin)
    china_data.plot(column='datavalue', cmap=cmap, linewidth=0.5, ax=axx, alpha=1, norm = norm)
    sm = plt.cm.ScalarMappable(norm, cmap)
    cax = fig.add_axes([ax.get_position().x0+axx.get_position().width, ax.get_position().y1+0.04, 0.008, axx.get_position().height-0.02])
    cbar = fig.colorbar(sm,cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=12) 
    cbar.set_ticks(bins)
    cbar.set_ticklabels(bins)

chinap_shp = '../data/China_shp/china_province.shp'
NH_shp = '../data/China_shp/NH_boundary.shp'
TXA_shp = '../data/China_shp/TW&HK&AM_province.shp'

(pro_wr_km3,pro_pre_km3,pro_GDP,pro_pop,pro_food,pro_agr,pro_ind,pro_ser,pro_wr_km3_sort,pro_name_list_sort_s,pro_pre_km3_sort,
sorted_index_S,pro_pop_sort,pro_food_sort) = social_economic_data_sort()

fig= plt.figure(figsize=(12, 12), dpi=300)
# water resource plot
ax1 = fig.add_axes([0, 0, 0.3, 0.8])
ax1.yaxis.tick_left()
ax1.set_xlim(0,450)
barh_plot(ax1,pro_name_list_sort_s,pro_wr_km3_sort,'Water resource','c','lower left','km\u00B3',[0, 100, 200, 300, 400])
China_map_plot(ax1,chinap_shp,NH_shp,TXA_shp,pro_wr_km3,[0, 100, 200, 300, 400, 500],'YlGnBu','(a) Water resource')
ax1.xaxis.set_label_coords(0.05, 1.06)

# precipitation plot
ax2 = fig.add_axes([0.3,0,0.3,0.8])
ax2.get_yaxis().set_visible(False)
ax2.set_xlim(0,850)
barh_plot(ax2,pro_name_list_sort_s,pro_pre_km3_sort,'Precipitation','#82B0D2','lower right','km\u00B3',[0, 150, 300, 450, 600, 750])
China_map_plot(ax2,chinap_shp,NH_shp,TXA_shp,pro_pre_km3,[0, 150, 300, 450, 600, 750],'PuBuGn','(b) Precipitation')
ax2.xaxis.set_label_coords(0.05, 1.06)

# GDP plot
def ax3_plot(ax3):
    labellist2 = ['Agriculture', 'Industry', 'Services']
    colorlist2 = ['#8ECFC9', '#FFBE7A', '#FA7F6F']
    ET_embodied_GDP = np.concatenate((pro_agr.reshape(31,1),pro_ind.reshape(31,1),pro_ser.reshape(31,1)),axis=1)
    sum_values2 = np.zeros(len(pro_name_list_sort_s))
    for i in range(ET_embodied_GDP.shape[1]):
        sorted_data2 = ET_embodied_GDP[:,i][sorted_index_S]
        ax3.barh(pro_name_list_sort_s, sorted_data2, label=labellist2[i], color=colorlist2[i], alpha=1, left=sum_values2, height=0.8)
        sum_values2 += sorted_data2

    ax3.xaxis.tick_top()
    ax3.margins(y=0.0)
    ax3.get_yaxis().set_visible(False)
    ax3.invert_yaxis()
    ax3.set_xlim(0,7.2)
    ax3.legend(loc='lower right', prop={'size': 16}, frameon=False)
    ax3.set_xlabel('trillion RMB',fontsize = 16)
    ax3.xaxis.set_label_coords(0.15, 1.06)
    ax3.set_xticks([0,1.5,3,4.5,6])
    ax3.set_xticklabels([0,1.5,3,4.5,6])
    ax3.tick_params(axis='both', labelsize=16)

ax3 = fig.add_axes([0.6,0,0.3,0.8])
ax3_plot(ax3)
China_map_plot(ax3,chinap_shp,NH_shp,TXA_shp,pro_GDP,[0,1.5,3,4.5,6,7.5],'YlOrRd','(c) Total GDP')

# population plot
ax4 = fig.add_axes([0.9,0,0.3,0.8])
ax4.get_yaxis().set_visible(False)
ax4.set_xlim(0,120)
barh_plot(ax4,pro_name_list_sort_s,pro_pop_sort,'Population','#BEB8DC','lower right','million persons',[0, 25, 50, 75, 100])
China_map_plot(ax4,chinap_shp,NH_shp,TXA_shp,pro_pop,[0, 25, 50, 75, 100, 125],'PuOr','(d) Population')

# food production plot
ax5 = fig.add_axes([1.2,0,0.3,0.8])
ax5.get_yaxis().set_visible(False)
ax5.set_xlim(0,70)
barh_plot(ax5,pro_name_list_sort_s,pro_food_sort,'Food production','#54B345','lower right','million ton',[0, 15, 30, 45, 60])
China_map_plot(ax5,chinap_shp,NH_shp,TXA_shp,pro_food,[0, 15, 30, 45, 60],'YlGn','(e) Food production')

fig.savefig('../figure/figure_A3.svg', dpi = 300, bbox_inches='tight')