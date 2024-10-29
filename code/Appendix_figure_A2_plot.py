import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader

# PRR, DSR, DPR calculate
def data_cal_sort():
    data_pre_relative = pd.read_csv('../data/prec_source_zonal0601.csv')
    pro_ETpre_relative = np.array(data_pre_relative.iloc[0:31,1:32])
    data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
    ET_data = pd.read_csv('../data/et_zonal_ERA5.csv')
    pro_name_list = data_h['province'].tolist()
    # unit km3
    pro_pre_km3_total = np.array(data_h['average_pre(108m3)']).reshape((1,31))/10
    
    # PRR
    self_recycling = np.diag(pro_ETpre_relative)
    # pre formed ET (km3)
    pro_ETpre_km3 = pro_ETpre_relative * pro_pre_km3_total
    # domestic pre by ET
    pro_ETpre_km3_sum  = pro_ETpre_km3.sum(axis=1)
    ET_in_dom_ratio = pro_ETpre_km3_sum / (ET_data['et'].values * data_h['area(km2)'].values / 1000000)
    # pre from domestic ET
    pro_ETpre_km3_dom = pro_ETpre_km3.sum(axis=0) 
    # pre from domestic ET / total pre
    pro_ETpre_km3_dom_ratio  = pro_ETpre_km3_dom / (data_h['average_pre(108m3)'].values/10)

    # sort
    sorted_index_DPR = np.argsort(ET_in_dom_ratio)[::-1]
    # pre from domestic ET
    DPR_sort = ET_in_dom_ratio[sorted_index_DPR]
    pro_name_list_DPR = np.array(pro_name_list)[sorted_index_DPR]
    DSR_sort = pro_ETpre_km3_dom_ratio[np.argsort(pro_ETpre_km3_dom_ratio)[::-1]]
    pro_name_list_DSR = np.array(pro_name_list)[np.argsort(pro_ETpre_km3_dom_ratio)[::-1]]
    PRR_sort = self_recycling[np.argsort(self_recycling)[::-1]]
    pro_name_list_PRR = np.array(pro_name_list)[np.argsort(self_recycling)[::-1]]

    return ET_in_dom_ratio,pro_ETpre_km3_dom_ratio,self_recycling,DPR_sort,pro_name_list_DPR,DSR_sort,pro_name_list_DSR,PRR_sort,pro_name_list_PRR

ET_in_dom_ratio,pro_ETpre_km3_dom_ratio,self_recycling,DPR_sort,pro_name_list_DPR,DSR_sort,pro_name_list_DSR,PRR_sort,pro_name_list_PRR = data_cal_sort()

# bar plot
def bar_map_sort(ax1,pro_name_list,ratio_data_sort,ratio_data,label,ylabel,unit,text,move1,
                 bins1,barcolor,ymin,cmap_name,bar_ymin,ticks1):
    
    ax1.bar(pro_name_list, ratio_data_sort, label=label,color=barcolor,width=0.75)
    ax1.legend(bbox_to_anchor=(0.25, 0.9), loc='center', prop={'size': 16}, frameon=False)
    ax1.margins(x=0.01)
    ax1.set_xticks(range(len(pro_name_list)))
    ax1.set_xticklabels(pro_name_list, rotation=90,fontsize = 16)
    ax1.set_ylabel(ylabel,fontsize=16)
    ax1.set_yticks(bins1)
    ax1.tick_params(axis='both', labelsize=16)
    # unit
    ax1.text(0,1.05,unit,transform=ax1.transAxes, fontdict={'size': '18', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    # num
    ax1.text(-0.08,1.05,text,transform=ax1.transAxes, fontdict={'size': '18', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    
    # data annotation
    data_value = ratio_data_sort.copy()
    for i in range(len(pro_name_list)):
        color = 'darkred'
        data_value[i] = ratio_data_sort[i]+move1
        text_ratio = round(ratio_data_sort[i],3)
        formatted_text = f"{text_ratio:.3f}".lstrip('0')
        ax1.text(i, data_value[i], formatted_text , ha='center',va='center',color=color,fontsize=12)

    chinap_shp = '../data/China_shp/china_province.shp'
    NH_shp = '../data/China_shp/NH_boundary.shp'
    TXA_shp = '../data/China_shp/TW&HK&AM_province.shp'

    ax11 = fig.add_axes([0.6,ymin,0.2,0.22],projection=ccrs.PlateCarree(),frameon=False)
    ax11.add_geometries(shpreader.Reader(chinap_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', lw=0.5,alpha=1)
    ax11.add_geometries(shpreader.Reader(NH_shp).geometries(),\
                        crs = ccrs.PlateCarree(),facecolor = 'None',lw = 0.5,alpha = 1)
    ax11.add_geometries(shpreader.Reader(TXA_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', hatch='\\\\\\\\',lw=0.5,alpha=1) 
    ax11.set_extent([70, 140, 2, 50],ccrs.Geodetic())

    china_data1 = chinap_code(chinap_shp,pd.DataFrame(ratio_data))
    nbin1 = len(bins1) - 1
    cmap1 = plt.cm.get_cmap(cmap_name, nbin1)
    norm1 = mcolors.BoundaryNorm(bins1, nbin1)
    china_data1.plot(column='datavalue', cmap=cmap1, linewidth=0.5, ax=ax11, alpha=1, norm=norm1)
    sm1 = plt.cm.ScalarMappable(norm1, cmap1)
    cax1 = fig.add_axes([0.8,bar_ymin,0.008,0.08])
    cbar1 = fig.colorbar(sm1,cax=cax1, orientation='vertical')
    cbar1.ax.tick_params(labelsize=14) 
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
fig.subplots_adjust(hspace=0.45)

ax1 = fig.add_subplot(311)
bar_map_sort(ax1,pro_name_list_DPR,DPR_sort,ET_in_dom_ratio,'Domestic precipitation ratio (DPR)','','','(a)',0.02,[0,0.2,0.4,0.6,0.8,1],'yellowgreen',
             0.69,'YlGn',0.79,['0','0.2','0.4','0.6','0.8','1'])
ax2 = fig.add_subplot(312)
bar_map_sort(ax2,pro_name_list_DSR,DSR_sort,pro_ETpre_km3_dom_ratio,'Domestic source ratio (DSR)','','','(b)',0.02,[0,0.2,0.4,0.6,0.8,1],'seagreen',
             0.4,'YlGnBu',0.5,['0','0.2','0.4','0.6','0.8','1'])
ax3 = fig.add_subplot(313)
bar_map_sort(ax3,pro_name_list_PRR,PRR_sort,self_recycling,'Precipitation recycling ratio (PRR)','','','(c)',0.01,[0,0.1,0.2,0.3,0.4],'lightseagreen',
             0.11,'PuBuGn',0.21,['0','0.1','0.2','0.3','0.4'])

fig.savefig('../figure/figure_A2.svg', dpi = 300)