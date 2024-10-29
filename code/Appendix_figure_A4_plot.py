import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader

data = pd.read_csv('../data/et_zonal_ERA5.csv')
pro_name = data['name'].values
ET_data = data['et'].values
data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
area = data_h['area(km2)'].values
# ET volume
ET_km3 = ET_data * area / 1000000

sorted_indices = np.argsort(ET_km3)[::-1]
sorted_data = ET_km3[sorted_indices]
sorted_name = pro_name[sorted_indices]

def bar_map_sort(ax1,data_sort,pro_name_list_sort,label,ylabel,unit,text,
                 ymin,data,bar_ymin,bins1,ticks1,barcolor,cmap_name):
    
    ax1.bar(pro_name_list_sort, data_sort, label=label,color=barcolor,width=0.75)
    ax1.legend(bbox_to_anchor=(0.2, 0.88), loc='center', prop={'size': 16}, frameon=False)
    ax1.margins(x=0.01)
    ax1.set_xticks(range(len(pro_name_list_sort)))
    ax1.set_xticklabels(pro_name_list_sort, rotation=90,fontsize = 16)
    ax1.set_ylabel(ylabel,fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    # unit
    ax1.text(-0.02,1.05,unit,transform=ax1.transAxes, fontdict={'size': '16', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    # num
    ax1.text(-0.08,1.05,text,transform=ax1.transAxes, fontdict={'size': '16', 'color': 'black'},
                horizontalalignment='center', verticalalignment='center')
    
    # data annotation
    data_value = data_sort.copy()
    for i in range(data_sort.shape[0]):
        data_value[i] = data_sort[i]+10
        ax.text(i, data_value[i], int(round(data_sort[i],0)), ha='center',va='center',color='black',fontsize=10)

    chinap_shp = '../data/China_shp/china_province.shp'
    NH_shp = '../data/China_shp/NH_boundary.shp'
    TXA_shp = '../data/China_shp/TW&HK&AM_province.shp'

    ax11 = fig.add_axes([0.3,ymin,0.6,0.66],projection=ccrs.PlateCarree(),frameon=False)
    ax11.add_geometries(shpreader.Reader(chinap_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', lw=0.5,alpha=1)
    ax11.add_geometries(shpreader.Reader(NH_shp).geometries(),\
                        crs = ccrs.PlateCarree(),facecolor = 'None',lw = 0.5,alpha = 1)
    ax11.add_geometries(shpreader.Reader(TXA_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', hatch='\\\\\\\\',lw=0.5,alpha=1) 
    ax11.set_extent([70, 140, 2, 50],ccrs.Geodetic())

    china_data1 = chinap_code(chinap_shp,pd.DataFrame(data))
    nbin1 = len(bins1) - 1
    cmap1 = plt.cm.get_cmap(cmap_name, nbin1)
    norm1 = mcolors.BoundaryNorm(bins1, nbin1)
    china_data1.plot(column='datavalue', cmap=cmap1, linewidth=0.5, ax=ax11, alpha=1, norm=norm1)
    sm1 = plt.cm.ScalarMappable(norm1, cmap1)
    cax1 = fig.add_axes([0.75,bar_ymin,0.015,0.4])
    cbar1 = fig.colorbar(sm1,cax=cax1, orientation='vertical')
    cbar1.ax.tick_params(labelsize=12) 
    cbar1.set_ticks(bins1)
    cbar1.set_ticklabels(ticks1)

# china province shp
def chinap_code(shp_path,data_value):
    shp_data = gpd.read_file(shp_path)
    pro_info= pd.read_csv('../data/china_province_center_point.csv')
    pro_code = pro_info.iloc[:,0]
    data_df = pd.concat([pro_code,data_value],axis = 1,ignore_index = True)
    data_df = data_df.rename(columns={0: "省代码",1: "datavalue"})
    pic_data = shp_data.merge(data_df, on='省代码')
    return pic_data

fig = plt.figure(figsize=(16, 6),dpi=300)
ax = fig.add_subplot(111)

bar_map_sort(ax, sorted_data, sorted_name,'Evapotranspiration','Evapotranspiration','km\u00B3','',
                 0.2,ET_km3,0.4,[0,100,200,300,400,500],[0,100,200,300,400,500],'tab:blue','GnBu')

fig.savefig('../figure/figure_A4.svg', dpi = 300, bbox_inches='tight')