import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.io.shapereader as shpreader
import math
import matplotlib.colors as mcolors

# get provincial list with x and y information
def get_pro_list():
    # contains location (x,y) of each province in China
    pro_xy=pd.read_csv('../data/china_province_center_point.csv')
    # expand origional province list to include xy
    pro_list=pd.read_csv('../data/china_province_list.csv')
    pro_xy.rename(columns={'省代码':'id'},inplace=True)
    pro_list = pro_list.merge(pro_xy,on='id').set_index('id')
    return pro_list

# calculate moisture transfer direction at provincial level along x and y direction
# type: source means the direction at source region
def calculate_moisture_transfer_direction(type='source'):
    pro_list=get_pro_list()
    if type=='source':
        dpc=pd.read_csv('../data/pre_108m3_target_tp_0601.csv').set_index('name')
    # create x and y direction ratio following the format of precipitation contribution
    x_div=dpc.copy()
    y_div=dpc.copy()
    # Calculate the component raio of transfer between provinces matrics along x and y direction
    # this ratio is to scale z value to x and y directions
    # ignore local to local transfer
    for source in dpc.columns:
        for target in dpc.index:
            source_xy = pro_list.loc[pro_list['name']==source.split('_', 1)[0]][['xcoord','ycoord']].values
            target_xy = pro_list.loc[pro_list['name']==target][['xcoord','ycoord']].values
            diff_xy = (target_xy-source_xy).squeeze()
            x_div.loc[target,source] = diff_xy[0]/((diff_xy[0]**2+diff_xy[1]**2)**0.5)
            y_div.loc[target,source] = diff_xy[1]/((diff_xy[0]**2+diff_xy[1]**2)**0.5)
    return x_div, y_div

def calculate_angle(x, y):
    anglead = math.atan(y / x) 
    angle_deg = math.degrees(anglead)
    return angle_deg

def chinap_code(shp_path,data_value):
    shp_data = gpd.read_file(shp_path)
    pro_info= pd.read_csv('../data/china_province_center_point.csv')
    pro_code = pro_info.iloc[:,0]
    data_df = pd.concat([pro_code,data_value],axis = 1,ignore_index = True)
    data_df = data_df.rename(columns={0: "省代码",1: "datavalue"})
    pic_data = shp_data.merge(data_df, on='省代码')
    return pic_data

x_div,y_div = calculate_moisture_transfer_direction(type='source')
pro_list = get_pro_list()
pro_list = pro_list.iloc[0:31,:]
dpc=pd.read_csv('../data/pre_108m3_target_tp_0601.csv').set_index('name')

x_sum = np.array((x_div*dpc).sum()).reshape(31,1)
y_sum = np.array((y_div*dpc).sum()).reshape(31,1)

# flow of total provinces
angle = calculate_angle(x_sum.sum(), y_sum.sum())

# calculate ET form pre in domestic ratio
data_pre_relative = pd.read_csv('../data/prec_source_zonal0601.csv')
pro_ETpre_relative = np.array(data_pre_relative.iloc[0:31,1:32])
data_h = pd.read_csv('../data/GDP&waterwithdrawal.csv')
ET_data = pd.read_csv('../data/et_zonal_ERA5.csv')
# unit km3
pro_pre_km3_total = np.array(data_h['average_pre(108m3)']).reshape((1,31))/10
# pre formed ET (km3)
pro_ETpre_km3 = pro_ETpre_relative * pro_pre_km3_total
# domestic pre by ET
pro_ETpre_km3_sum  =pro_ETpre_km3.sum(axis=1)
ET_in_dom_ratio = pro_ETpre_km3_sum / (ET_data['et'].values * data_h['area(km2)'].values / 1000000)
ET_in_dom_ratio = pd.DataFrame(ET_in_dom_ratio)

# green water flow plot
def flow_plot(ax,axx,chinap_shp,NH_shp,TXA_shp):
    ax.set_extent([72.5, 135.5, 18, 52], ccrs.Geodetic())
    ax.add_geometries(shpreader.Reader(chinap_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', lw=0.5,alpha=1)
    # Nanhai & taiwan, hongkong, Macao
    ax.add_geometries(shpreader.Reader(NH_shp).geometries(),\
                        crs = ccrs.PlateCarree(),facecolor = 'None',lw = 0.5,alpha = 1)
    ax.add_geometries(shpreader.Reader(TXA_shp).geometries(),crs=ccrs.PlateCarree(),\
                        facecolor='none',hatch='\\\\\\',lw=0.5,alpha = 1)

    china_data = chinap_code(chinap_shp,ET_in_dom_ratio)
    bins = [0.2, 0.4, 0.6, 0.8, 1]
    nbin = len(bins) - 1
    cmap = cm.get_cmap('YlGnBu', nbin)
    norm = mcolors.BoundaryNorm(bins, nbin)
    china_data.plot(column='datavalue', cmap=cmap, linewidth=0.5, ax=ax, alpha=0.8, norm = norm)
    sm = plt.cm.ScalarMappable(norm, cmap)
    cax = fig.add_axes([ax.get_position().xmin+1/2*ax.get_position().width-0.11,ax.get_position().ymax-0.13,0.2,0.015])
    cbar = fig.colorbar(sm,cax=cax, orientation='horizontal', alpha = 0.8)
    cbar.ax.tick_params(labelsize=10) 
    cbar.set_ticks(bins)
    cbar.set_ticklabels(bins)
    cbar.set_label('Domestic precipitation ratio (DPR)')
    cbar.ax.xaxis.set_label_position('top')

    # flow quiver
    ax.quiver(pro_list['xcoord'], pro_list['ycoord'], (x_div*dpc).sum(), (y_div*dpc).sum(), transform=ccrs.PlateCarree(), width =0.005, color='darkgreen')
    # province center point
    ax.scatter(pro_list['xcoord'], pro_list['ycoord'], c= 'darkred', s=20)

    axx.add_geometries(shpreader.Reader(chinap_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', lw=0.5,alpha=1)
    axx.add_geometries(shpreader.Reader(NH_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', lw=0.5,alpha=1)
    axx.add_geometries(shpreader.Reader(TXA_shp).geometries(),crs=ccrs.PlateCarree(),facecolor='none', hatch='\\\\\\',lw=0.5,alpha=1) 
    axx.set_extent([105, 125, 2, 25])
    china_data.plot(column='datavalue', cmap=cmap, linewidth=0.5, ax=axx, alpha=0.8, norm = norm)

    # RGB of the color bar
    colors_array = np.zeros((nbin, 3), dtype=int)
    for i in range(nbin):
        color = cmap(i / (nbin - 1))
        colors_array[i] = np.array(color[:3]) * 255

    return colors_array

# lower left plot
def arrow_total(ax, angle):
    length = 0.05
    anglead = angle * (3.14159 / 180)
    end_x = length * math.cos(anglead)
    end_y = length * math.sin(anglead)
    ax.arrow(0, 0, end_x, end_y, head_width=0.01, head_length=0.02, fc='darkgreen', ec='darkgreen')
    ax.arrow(-0.05, 0, 0.1, 0, head_width=0.01, head_length=0.02, fc='black', ec='black')
    ax.arrow(0, -0.05, 0, 0.1, head_width=0.01, head_length=0.02, fc='black', ec='black')
    ax.text(0.075, 0, 'E', fontsize=10, horizontalalignment='center', verticalalignment='center')
    ax.text(0, 0.08, 'N', fontsize=10, horizontalalignment='center', verticalalignment='center')
    ax.text(end_x+0.02, end_y+0.02, f'{int(round(angle,0))}\u00B0', color = 'darkgreen',fontsize=10, horizontalalignment='center', verticalalignment='center')
    ax.axis('off')

# upper left plot
def quiver_explain(ax, colors_array, xinjiang_shp):
    # schematic quiver plot
    def quiver_data_single(ax,a,b,scale):
        xj_o_dpc = dpc.iloc[a,b]
        xj_o_x_div = x_div.iloc[a,b]
        xj_o_y_div = y_div.iloc[a,b]
        xj_o_x_sum = xj_o_x_div * xj_o_dpc
        xj_o_y_sum = xj_o_y_div * xj_o_dpc
        Q3 = ax.quiver(xj_xy['xcoord'], xj_xy['ycoord'], xj_o_x_sum, xj_o_y_sum, transform=ccrs.PlateCarree(), 
                width =0.015, headwidth = 4,scale = scale, color='olivedrab', zorder=9)
        return Q3

    # xinjiang's green water flow
    xj_x_sum = (x_div*dpc).sum().iloc[-1]
    xj_y_sum = (y_div*dpc).sum().iloc[-1]

    # xinjiang shp plot
    xj_xy = pro_list[pro_list['name'] == 'Xinjiang']
    ax.set_extent([72, 98, 33, 50], crs=ccrs.PlateCarree())
    ax.add_geometries(shpreader.Reader(xinjiang_shp).geometries(), crs=ccrs.PlateCarree(), 
                      facecolor=(colors_array[1,0]/255,colors_array[1,1]/255,colors_array[1,2]/255), lw=1, alpha=0.8)
    ax.scatter(xj_xy['xcoord'], xj_xy['ycoord'], c= 'darkred', s=20, zorder=11)

    # schematic quiver
    Q1 = ax.quiver(xj_xy['xcoord'], xj_xy['ycoord'], -50, 30, transform=ccrs.PlateCarree(), width =0.015, headwidth = 4, scale = 200, color='coral',zorder=9)
    Q2 = ax.quiver(xj_xy['xcoord'], xj_xy['ycoord'], xj_x_sum, xj_y_sum, transform=ccrs.PlateCarree(), width =0.015, headwidth = 4, scale = 3000, color='darkgreen', zorder=10)
    Q3 = quiver_data_single(ax,4,-1,500)
    quiver_data_single(ax,24,-1,3)
    quiver_data_single(ax,25,-1,1000)
    ax.quiverkey(Q1, 1, 1 ,32 ,'Abroad', labelpos = 'E',fontproperties={'size': 8})
    ax.quiverkey(Q3, 1, 0.9 ,80 ,'Domestic', labelpos = 'E',fontproperties={'size': 8})
    ax.quiverkey(Q2, 1, 0.8 ,480 ,'Domestic flow synthesis', labelpos = 'E',fontproperties={'size': 8})
    ax.text(90,42.5,'Neimeng',fontsize='8')
    ax.text(87.5,35.5,'Yunnan',fontsize='8')
    ax.text(81.5,34,'Xizang',fontsize='8')

    # no boundary
    for spine in ax.spines.values():
        spine.set_color('none')
    ax.set_facecolor('none')

chinap_shp = '../data/China_shp/china_province.shp'
NH_shp = '../data/China_shp/NH_boundary.shp'
TXA_shp = '../data/China_shp/TW&HK&AM_province.shp'
xinjiang_shp = '../data/China_shp/xinjiang_g.shp'

fig = plt.figure(figsize=(12, 8),dpi=300)
ax = fig.add_axes([0,0,0.6,0.8],projection=ccrs.PlateCarree())
ax2 = fig.add_axes([ax.get_position().xmax-0.08, ax.get_position().ymin+0.1, 0.12, 0.165], projection=ccrs.PlateCarree())
ax3 = ax.inset_axes([0.05, 0.03, 0.2, 0.2], transform=ax.transAxes)
ax4 = fig.add_axes([ax.get_position().xmin-0.26, ax.get_position().ymax-0.25, 0.6, 0.15], projection=ccrs.PlateCarree())

colors_array = flow_plot(ax,ax2,chinap_shp,NH_shp,TXA_shp)
arrow_total(ax3,angle)
quiver_explain(ax4,colors_array,xinjiang_shp)

fig.savefig('../figure/figure_3.svg', dpi=300, bbox_inches='tight')
