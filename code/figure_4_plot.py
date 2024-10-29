import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader

# socio-economic effects of green water flow in each source province
def social_economic_data_sort():

    data_preelative = pd.read_csv('../data/prec_source_zonal0601.csv')
    pro_ETpreelative = data_preelative.iloc[0:31,1:32].values
    data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
    pro_name_list = data_h['province'].tolist()

    # unit km3
    pro_ETwr_km3 = pro_ETpreelative * data_h['average_water_resources(108m3)'].values.reshape((1,31))/10
    pro_ETwr_km3_sum = pro_ETwr_km3.sum(axis=1)

    # unit trillion rmb
    pro_ETGDP = pro_ETpreelative * data_h['sum_GDP_2020RMB(108)'].values.reshape((1,31))/10000
    pro_ETGDP_sum = pro_ETGDP.sum(axis=1)

    pro_ETagr = pro_ETpreelative * data_h['agr_GDP_2020RMB(108)'].values.reshape((1,31))/10000
    pro_ETagr_sum = pro_ETagr.sum(axis=1)
    pro_ETind = pro_ETpreelative * data_h['ind_GDP_2020RMB(108)'].values.reshape((1,31))/10000
    pro_ETind_sum = pro_ETind.sum(axis=1)
    pro_ETser = pro_ETpreelative * data_h['ser_GDP_2020RMB(108)'].values.reshape((1,31))/10000
    pro_ETser_sum = pro_ETser.sum(axis=1)

    # unit million pop
    pro_ETpop = pro_ETpreelative * data_h['population(wanren)'].values.reshape((1,31))/100
    pro_ETpop_sum = pro_ETpop.sum(axis=1)

    # unit million ton
    pro_ETfood = pro_ETpreelative * data_h['food_production(wan_t)'].values.reshape((1,31))/100
    pro_ETfood_sum = pro_ETfood.sum(axis=1)

    # sort
    sorted_index_S = np.argsort(pro_ETwr_km3_sum)[::-1]
    pro_ETwr_km3_sum_sort = pro_ETwr_km3_sum[sorted_index_S]
    pro_name_list_sort_s = np.array(pro_name_list)[sorted_index_S]

    pro_ETpop_sum_sort = pro_ETpop_sum[sorted_index_S]

    pro_ETfood_sum_sort = pro_ETfood_sum[sorted_index_S]

    return (pro_ETwr_km3_sum,pro_ETGDP_sum,pro_ETpop_sum,pro_ETfood_sum,pro_ETagr_sum,pro_ETind_sum,
            pro_ETser_sum,pro_ETwr_km3_sum_sort,pro_name_list_sort_s,
            sorted_index_S,pro_ETpop_sum_sort,pro_ETfood_sum_sort)

# china province shp
def chinap_code(shp_path,data_value):
    shp_data = gpd.read_file(shp_path)
    pro_info = pd.read_csv('../data/china_province_center_point.csv')
    data_df = pd.concat([pro_info.iloc[:,0],data_value],axis = 1,ignore_index = True)
    data_df = data_df.rename(columns={0: "省代码",1: "datavalue"})
    pic_data = shp_data.merge(data_df, on='省代码')
    return pic_data

# water resource, population and food production of green water in source provinces plot
def barh_plot(ax,pro_name_list_sort,data_sort,label,color,legloc,xlabel):
    ax.barh(np.arange(len(pro_name_list_sort)), data_sort, label=label,height=0.8,color=color)
    ax.xaxis.tick_top()
    ax.margins(y=0.0)
    ax.set_yticks(range(len(pro_name_list_sort)))
    ax.set_yticklabels(pro_name_list_sort_s.tolist())
    ax.invert_yaxis()
    ax.legend(loc=legloc, prop={'size': 16}, frameon=False)
    ax.set_xlabel(xlabel,fontsize = 16)
    ax.xaxis.set_label_coords(0.15, 1.06)
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

(pro_ETwr_km3_sum,pro_ETGDP_sum,pro_ETpop_sum,pro_ETfood_sum,pro_ETagr_sum,pro_ETind_sum,pro_ETser_sum,
pro_ETwr_km3_sum_sort,pro_name_list_sort_s,sorted_index_S,pro_ETpop_sum_sort,pro_ETfood_sum_sort) = social_economic_data_sort()


fig= plt.figure(figsize=(12, 12), dpi=300)
# water resource value plot
ax1 = fig.add_axes([0, 0, 0.3, 0.8])
ax1.yaxis.tick_left()
ax1.set_xlim(0,240)
barh_plot(ax1,pro_name_list_sort_s,pro_ETwr_km3_sum_sort,'Water resource','#82B0D2','lower left','km\u00B3')
China_map_plot(ax1,chinap_shp,NH_shp,TXA_shp,pro_ETwr_km3_sum,[0, 50, 100, 150, 200],'YlGnBu','(a) Water resource')
ax1.xaxis.set_label_coords(0.05, 1.06)

# economic value of green water plot
def ax2_plot(ax2):
    labellist2 = ['Agriculture', 'Industry', 'Services']
    colorlist2 = ['#8ECFC9', '#FFBE7A', '#FA7F6F']
    ET_embodied_GDP = np.concatenate((pro_ETagr_sum.reshape(31,1),pro_ETind_sum.reshape(31,1),pro_ETser_sum.reshape(31,1)),axis=1)
    sum_values2 = np.zeros(len(pro_name_list_sort_s))
    for i in range(ET_embodied_GDP.shape[1]):
        sorted_data2 = ET_embodied_GDP[:,i][sorted_index_S]
        ax2.barh(pro_name_list_sort_s, sorted_data2, label=labellist2[i], color=colorlist2[i], alpha=1, left=sum_values2, height=0.8)
        sum_values2 += sorted_data2

    ax2.xaxis.tick_top()
    ax2.margins(y=0)
    ax2.get_yaxis().set_visible(False)
    ax2.invert_yaxis()
    ax2.set_xlim(0,2.4)
    ax2.legend(loc='lower center', prop={'size': 16}, frameon=False)
    ax2.set_xlabel('trillion CNY',fontsize = 16)
    ax2.xaxis.set_label_coords(0.15, 1.06)
    ax2.set_xticks([0,0.5,1,1.5,2])
    ax2.set_xticklabels([0,0.5,1,1.5,2])
    ax2.tick_params(axis='both', labelsize=16)

ax2 = fig.add_axes([0.3,0,0.3,0.8])
ax2_plot(ax2)
China_map_plot(ax2,chinap_shp,NH_shp,TXA_shp,pro_ETGDP_sum,[0, 0.5, 1, 1.5, 2, 2.5],'YlOrRd','(b) Total GDP')

# population value plot
ax3 = fig.add_axes([0.6,0,0.3,0.8])
ax3.get_yaxis().set_visible(False)
ax3.set_xlim(0,75)
barh_plot(ax3,pro_name_list_sort_s,pro_ETpop_sum_sort,'Population','#BEB8DC','lower center','million persons')
China_map_plot(ax3,chinap_shp,NH_shp,TXA_shp,pro_ETpop_sum,[0, 10, 20, 30, 40, 50, 60],'PuOr','(c) Population')

# food production value plot
ax4 = fig.add_axes([0.9,0,0.3,0.8])
ax4.get_yaxis().set_visible(False)
ax4.set_xlim(0,25)
barh_plot(ax4,pro_name_list_sort_s,pro_ETfood_sum_sort,'Food production','#54B345','lower left','million ton')
China_map_plot(ax4,chinap_shp,NH_shp,TXA_shp,pro_ETfood_sum,[0, 5, 10, 15, 20, 25],'YlGn','(d) Food production')

fig.savefig('../figure/figure_4.svg', dpi = 300, bbox_inches='tight')