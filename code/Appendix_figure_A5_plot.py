import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader

# calculate net socio-economic value of green water from source to sink provinces
def socio_net_data(data,unit_data,m1,m2):   
    data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
    pro_name_list = data_h['province'].values 
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
    # net socio-economic value of green water from source to sink provinces
    water_ECOvalue_net = water_ECOvalue_outerunit_sum-water_ECOvalue_innerunit_sum

    sorted_indices = np.argsort(water_ECOvalue_net)
    water_ECOvalue_net_sort = water_ECOvalue_net[sorted_indices]
    pro_name_list_sort = pro_name_list[sorted_indices]

    return water_ECOvalue_net,water_ECOvalue_net_sort,pro_name_list_sort

def bar_map_sort(ax1,water_ECOvalue_net_sort,pro_name_list_sort,label,ylabel,unit,text,move1,move2,
                 ymin,water_ECOvalue_net,bar_ymin,bins1,ticks1,barcolor,cmap_name):
    
    ax1.bar(pro_name_list_sort, water_ECOvalue_net_sort, label=label,color=barcolor,width=0.75)
    ax1.legend(bbox_to_anchor=(0.75, 0.9), loc='center', prop={'size': 16}, frameon=False)
    ax1.margins(x=0.01)
    ax1.set_xticks(range(len(pro_name_list_sort)))
    ax1.set_xticklabels(pro_name_list_sort, rotation=90,fontsize = 16)
    ax1.set_ylabel(ylabel,fontsize=16)
    ax1.set_yticks(bins1)
    ax1.tick_params(axis='both', labelsize=16)
    # unit
    ax1.text(0,1.05,unit,transform=ax1.transAxes, fontdict={'size': '16', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    # num
    ax1.text(-0.08,1.05,text,transform=ax1.transAxes, fontdict={'size': '16', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    
    # data annotation
    data_value = water_ECOvalue_net_sort.copy()
    for i in range(len(pro_name_list_sort)):
        if water_ECOvalue_net_sort[i]>0:
            color = 'darkred'
            data_value[i] = water_ECOvalue_net_sort[i]+move1
        else:
            color = 'black'
            data_value[i] = water_ECOvalue_net_sort[i]-move2
        ax1.text(i, data_value[i], round(water_ECOvalue_net_sort[i],2), ha='center',va='center',color=color,fontsize=9)

    chinap_shp = '../data/China_shp/china_province.shp'
    NH_shp = '../data/China_shp/NH_boundary.shp'
    TXA_shp = '../data/China_shp/TW&HK&AM_province.shp'

    ax11 = fig.add_axes([0.3,ymin,0.2,0.22],projection=ccrs.PlateCarree(),frameon=False)
    ax11.add_geometries(shpreader.Reader(chinap_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', lw=0.5,alpha=1)
    ax11.add_geometries(shpreader.Reader(NH_shp).geometries(),\
                        crs = ccrs.PlateCarree(),facecolor = 'None',lw = 0.5,alpha = 1)
    ax11.add_geometries(shpreader.Reader(TXA_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', hatch='\\\\\\\\',lw=0.5,alpha=1) 
    ax11.set_extent([70, 140, 2, 50],ccrs.Geodetic())

    china_data1 = chinap_code(chinap_shp,pd.DataFrame(water_ECOvalue_net))
    nbin1 = len(bins1) - 1
    cmap1 = plt.cm.get_cmap(cmap_name, nbin1)
    norm1 = mcolors.BoundaryNorm(bins1, nbin1)
    china_data1.plot(column='datavalue', cmap=cmap1, linewidth=0.5, ax=ax11, alpha=1, norm=norm1)
    sm1 = plt.cm.ScalarMappable(norm1, cmap1)
    cax1 = fig.add_axes([0.5,bar_ymin,0.008,0.08])
    cbar1 = fig.colorbar(sm1,cax=cax1, orientation='vertical')
    cbar1.ax.tick_params(labelsize=12) 
    cbar1.set_ticks(bins1)
    cbar1.set_ticklabels(ticks1)

def chinap_code(shp_path,data_value):
    shp_data = gpd.read_file(shp_path)
    pro_info= pd.read_csv('../data/china_province_center_point.csv')
    pro_code = pro_info.iloc[:,0]
    data_df = pd.concat([pro_code,data_value],axis = 1,ignore_index = True)
    data_df = data_df.rename(columns={0: "省代码",1: "datavalue"})
    pic_data = shp_data.merge(data_df, on='省代码')
    return pic_data

fig = plt.figure(figsize=(16, 19),dpi=300)
fig.subplots_adjust(hspace=0.48)

ax1 = fig.add_subplot(311)
water_ECOvalue_net,water_ECOvalue_net_sort,pro_name_list_sort = socio_net_data('sum_GDP_2020RMB(108)','per_unit_water_value(rmb_m3)',0.0001,0.0001)
bar_map_sort(ax1, water_ECOvalue_net_sort,pro_name_list_sort,'Net economic value','Net economic value','trillion RMB','(a)',
             0.025,0.035,0.7,water_ECOvalue_net,0.795,[-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
             ['-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75', '1'],'lightcoral','coolwarm')
ax2 = fig.add_subplot(312)
water_ECOvalue_net,water_ECOvalue_net_sort,pro_name_list_sort = socio_net_data('population(wanren)','per_unit_water_pop(pop_m3)',0.01,100)
bar_map_sort(ax2, water_ECOvalue_net_sort,pro_name_list_sort,'Net population value','Net population value','million persons','(b)',
             0.5,0.5,0.413,water_ECOvalue_net,0.505,[-15, -10, -5, 0, 5, 10, 15],
             ['-15', '-10','-5', '0', '5', '10', '15'],'olivedrab','PiYG')
ax3 = fig.add_subplot(313)
water_ECOvalue_net,water_ECOvalue_net_sort,pro_name_list_sort = socio_net_data('food_production(wan_t)','per_unit_water_food(ton_m3)',0.01,100)
bar_map_sort(ax3, water_ECOvalue_net_sort,pro_name_list_sort,'Net food production value','Net food production value','million ton','(c)',
             0.5,0.5,0.125,water_ECOvalue_net,0.218,[-15, -10, -5, 0, 5, 10, 15],
             ['-15', '-10', '-5', '0', '5', '10', '15'],'darkgoldenrod','BrBG')

fig.savefig('../figure/figure_A5.svg', dpi = 300)