import pandas as pd
import re, ast
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from googlemaps import Client
import random as rd
from matplotlib import interactive
interactive(True)
gmaps = Client('AIzaSyCo9Oys4uUzz1Bh2aUtEneXfOgHku7Ls2I')


#Colors
colors = ['bo','ro','yo']

#Get coordinates
def lookfor(plaz):
    result = gmaps.geocode(plaz)
    placemark = result[0]['geometry']['location']
    lat = placemark['lat']
    lng = placemark['lng']
    x_,y_ = m(lng,lat)
    final = [x_,y_]
    return final

#Mount map
def give_map_NA(res):
    m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64,
                               urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45,
                               lon_0=-95, resolution=res, area_thresh=10000)
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    m.drawmapboundary(ax=ax,fill_color='#b0e0e6')
    m.drawcoastlines(ax=ax,color='white',linewidth=0.1,)
    m.drawcountries(ax=ax,color='#006400')
    m.fillcontinents(ax=ax,color='#2e8b57',lake_color='#b0e0e6')
    return m

#Mount map UK.
def give_map_UK(res):
    m = Basemap(projection='stere',lon_0=-2.55,lat_0=55.0,lat_ts=55.0,
                        width=600000,height=1150000,
                        resolution=res,area_thresh=10000)
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    m.drawmapboundary(ax=ax,fill_color='#b0e0e6')
    m.drawcoastlines(ax=ax,color='white',linewidth=0.1,)
    m.drawcountries(ax=ax,color='#006400')
    m.fillcontinents(ax=ax,color='#2e8b57',lake_color='#b0e0e6')
    return m

#Mount map NE.
def give_map_NE(res):
    m = Basemap(llcrnrlon=-90, llcrnrlat=35, urcrnrlon=-64,
                               urcrnrlat=45, projection='lcc', lat_1=33, lat_2=45,
                               lon_0=-95, resolution=res, area_thresh=10000)
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    m.drawmapboundary(ax=ax,fill_color='#b0e0e6')
    m.drawcoastlines(ax=ax,color='white',linewidth=0.1,)
    m.drawcountries(ax=ax,color='#006400')
    m.fillcontinents(ax=ax,color='#2e8b57',lake_color='#b0e0e6')
    return m

#Plot map
def plot_SA(color,data,zone):
    x,y = [],[]
    if zone == 'NA':
        m = give_map_NA('c')
    elif zone == 'UK':
        m = give_map_UK('c')
    elif zone == 'NE':
        m = give_map_NE('c')
    for i in data:
        try:
            x.append(float(i[0]))
            y.append(float(i[1]))
        except:
            pass
    x_,y_ = m(x,y)
    print '\nData with n ='+str(len(x_))
    m.plot(x_, y_, color, markersize=5)
    plt.show()

#Plot group
def group_plot_SA(colors,data,zone):
    if zone == 'NA':
        m = give_map_NA('i')
    elif zone == 'UK':
        m = give_map_UK('i')
    elif zone == 'NE':
        m = give_map_NE('i')
    for j in range(len(data)):
        x,y = [],[]
        for i in data[j]:
            try:
                x.append(float(i[0]))
                y.append(float(i[1]))
            except:
                pass
        x_,y_ = m(x,y)
        print 'Color '+str(colors[j])+' has n ='+str(len(x_))
        m.plot(x_, y_, colors[j], markersize=5)
    if zone == 'NA': plt.title('Twitter opinions in the United States')
    if zone == 'UK': plt.title('Twitter opinions in the United Kingdom')
    if zone == 'NE': plt.title('Twitter opinions in the North Eastern United States')
    plt.show()


#target 0.day,1.pol,2.coord,3.geo,4.place,5.place_home           #index day. 0=pol, 1=coord
def get_space(initial):
    data = {}

    for j in initial:
        initial_df = pd.load(j)

        target = pd.DataFrame(initial_df.time)
        target2 = target.join(initial_df[['polarity','coords','geo','place','place_home']])
        for i in target2.values:
	                if rd.randint(0,20) == 1:
					print j,i[0]
	                valid = False
	                final = 'None'
	                for k in range(len(i)):
	                    if k == 0 or k == 1:
	                        pass
	                    else:
	                        if str(i[k]) != 'nan' and str(i[k]) != 'None':
	                            if k == 2 or k == 3:
	                               try:
	                                dicti = ast.literal_eval(str(i[k]))
	                                coords = dicti['coordinates']
	                                if coords == [0.0, 0.0]:
	                                    valid = False
	                                    pass
	                                else:
	                                    final = coords.reverse()
	                                    valid = True
	                                    break
	                               except:
	                                pass
	                            elif k == 4:
	                                place = re.split('],',str(i[4]))
	                                final = re.split(', ',str(place[0][3:]))
	                                valid = True
	                                break
	                            elif k == 100:
	                                if ',' in str(i[k]):
	                                    print str(i[k])
	                                    break
	                                    try:
	                                        final  = lookfor(str(i[k]))
	                                        valid = True
	                                        print 'WORKED!!!!!!!!!!'
	                                        break
	                                    except:
	                                        valid = False
	                                        break		
	                        valid = False
	                if valid == True:
	                    place = re.split('],',str(i[4]))
	                    data.update({str(i[0]):[i[1],i[2],i[3],place[0][3:],i[5],final]})
        print j
    data_df = pd.DataFrame.from_dict(data,orient='index')
    data_df.columns = ['pol','coord','geo','place','home','final']
    data_df = data_df.sort()
    return data_df


# ##Alien Isolation
#alien_df = get_space(['alien_isolation_df'])
#data_df = alien_df
#groups = [data_df.final[data_df.pol > 0],data_df.final[data_df.pol < 0],data_df.final[data_df.pol == 0]]
#group_plot_SA(colors,groups,'NA')
#group_plot_SA(colors,groups,'UK')


# ##Dragon Age
#dragon_df = get_space(['dragon_age_df'])
#data_df = dragon_df
#groups = [data_df.final[data_df.pol > 0],data_df.final[data_df.pol < 0],data_df.final[data_df.pol == 0]]
#group_plot_SA(colors,groups,'NA')
#group_plot_SA(colors,groups,'UK')


# ##Call of Duty
#cod_df = get_space(['call_of_duty_df'])
#data_df = cod_df
#groups = [data_df.final[data_df.pol > 0],data_df.final[data_df.pol < 0],data_df.final[data_df.pol == 0]]
#group_plot_SA(colors,groups,'NA')
#group_plot_SA(colors,groups,'UK')


# ##Bayonetta 2
#bay_df = get_space(['bayonetta2_df'])
#data_df = bay_df
#groups = [data_df.final[data_df.pol > 0],data_df.final[data_df.pol < 0],data_df.final[data_df.pol == 0]]
#group_plot_SA(colors,groups,'NA')
#group_plot_SA(colors,groups,'UK')


# ##Hearthstone
#her_df = get_space(['hearthstone_df'])
#data_df = her_df
#groups = [data_df.final[data_df.pol > 0],data_df.final[data_df.pol < 0],data_df.final[data_df.pol == 0]]
#group_plot_SA(colors,groups,'NA')
#group_plot_SA(colors,groups,'UK')


# ##Fifa
#fifa_df = get_space(['fifa_15_df'])
# ##GTA

# ##Dying Light
#dl_df = get_space(['dying_light_df'])

# ##Titanfall
#titan_df = get_space(['titanfall_df'])
# all_df = ['dying_light_df','alien_isolation_df','bayonetta2_df','dragon_age_df','call_of_duty_df','fifa_15_df','gta_df','hearthstone_df','titanfall_df']
all_df = pd.load('all_df')
