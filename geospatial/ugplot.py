# -*- coding: utf-8 -*-
"""
Uganda-specific plotting functions.

Created on Tue May 20 18:46:51 2014

@author: John Quinn <john.quinn@one.un.org>
"""
import shapefile
import pylab as plt
import numpy as np
import os
import matplotlib 
import matplotlib.patheffects as path_effects
import cv2
from scipy.stats import multivariate_normal
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
    

geospatial_dir = os.path.dirname(__file__)
if geospatial_dir:
    geospatial_dir += os.sep
sf_districts = shapefile.Reader(geospatial_dir + "shapefiles/UGA_districts2010_simplified")
district_shapes = sf_districts.shapes()
sf_country = shapefile.Reader(geospatial_dir + "shapefiles/UGA_country")
country_shapes = sf_country.shapes()
sf_water = shapefile.Reader(geospatial_dir + "shapefiles/Ug_Waterbodies")
water_shapes = sf_water.shapes()
sf_roads = shapefile.Reader(geospatial_dir + "shapefiles/Uganda_Roads")
roads_shapes = sf_roads.shapes()
sf_reserves = shapefile.Reader(geospatial_dir + "shapefiles/protectedsites")
reserves_shapes = sf_reserves.shapes()

UG_min_lon = 29.571499
UG_max_lon = 35.000273
UG_min_lat = -1.47887
UG_max_lat = 4.234466

coords = np.array(country_shapes[0].points)
X_UGA = coords[:,0]
Y_UGA = coords[:,1] 
xlim_l = min(X_UGA) - .1
xlim_u = max(X_UGA) + .1
ylim_l = min(Y_UGA) - .1
ylim_u = max(Y_UGA) + .1

lm = None
rm = None
   
def plot_kde_density(X, y, resolution=40, bandwidth=.1, showplot=True):
    '''
    Plot kernel density estimate across the map given observations (y) at 
    specific locations (X)
    resolution: number of pixels per arc degree
    bandwidth: length scale in arc degrees
    showplot: if set to False, only calculate, don't plot
    '''

    width = resolution*(UG_max_lon - UG_min_lon)
    height = resolution*(UG_max_lat - UG_min_lat)
    x1, x2 = np.meshgrid(np.linspace(UG_min_lon, UG_max_lon, width),
                         np.linspace(UG_min_lat, UG_max_lat, height))
    xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T
    
    ypred = np.zeros((xx.shape[0]))
    for i in range(X.shape[0]):
        if not np.any(np.isnan(X[i,:])):
            rv = multivariate_normal(X[i,:], cov=np.eye(2)*bandwidth)
            ypred += rv.pdf(xx) * y[i]
    
    ypred = np.flipud(np.reshape(ypred,(height,width)))
    
    if showplot:
        plot_uganda_cropped_image(ypred)
    
    return ypred

    
def plot_country_outline(background_colour=(1,1,1), outline_colour=(0,0,0)):
    '''Draw the outline of Uganda'''
    margin = 1.5
    ax = plt.gca()
    ax.fill(np.hstack((xlim_l-margin, xlim_u+margin, xlim_u+margin, 
                       xlim_l-margin, xlim_l-margin, X_UGA)), 
            np.hstack((ylim_l-margin, ylim_l-margin, ylim_u+margin, 
                        ylim_u+margin, ylim_l-margin, Y_UGA)), 
            color=background_colour)
    ax.plot(X_UGA,Y_UGA,color=outline_colour)
    ax.axis('equal')  
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xlim(xlim_l, xlim_u)
    ax.set_ylim(ylim_l, ylim_u) 

def plot_single_district(district,outline_colour=(0,0,0),fill_colour=None):
    r = sf_districts.records()
    for i_record in range(len(r)):
        if r[i_record][5].lower() == district.lower():
            coords = np.array(district_shapes[i_record].points)
            X_district = coords[:,0]
            Y_district = coords[:,1]
            if district.lower()=='lamwo': # workaround for bad info in the official shapefile
                X_district = X_district[:-8]
                Y_district = Y_district[:-8]
            if district.lower()=='mukono':
                X_district = X_district[:-15]
                Y_district = Y_district[:-15]                
            ax = plt.gca()
            if fill_colour:
                ax.fill(X_district, Y_district, color=fill_colour)
            if outline_colour:
                ax.plot(X_district,Y_district,color=outline_colour)  

def annotate_district(district, text, label_colour='k', label_shadow='w', size=10,
                      voffset=0, hoffset=0, use_preset_offsets=False):
    '''Display text at the location of a named district'''
    r = sf_districts.records()
    for i_record in range(len(r)):
        if r[i_record][5].lower() == district.lower():
            coords = np.array(district_shapes[i_record].points)
            X_district = coords[:,0]
            Y_district = coords[:,1] 
            
            if use_preset_offsets:
                if district=='Kampala':
                    Y_district-=.15
                if district=='Wakiso':
                    Y_district-=.05                
                if district=='Mayuge':
                    X_district+=.05 
                if district=='Mukono':
                    Y_district+=.05
                if district=='Iganga':
                    Y_district+=.05
                if district=='Ngora':
                    Y_district+=.1
                if district=='Amuria':
                    Y_district+=.1
                if district=='Rukungiri':
                    Y_district+=.07
                if district=='Bundibugyo':
                    Y_district+=.1
                if district=='Kyenjojo':
                    Y_district+=.08
                if district=='Pader':
                    X_district+=.1
                if district=='Kole':
                    Y_district-=.1
                if district=='Kasese':
                    Y_district-=.07
                if district=='Mayuge':
                    Y_district-=.1
                if district=='Kabale':
                    Y_district-=.1
                if district=='Kalangala':
                    Y_district+=.1
            
            Y_district += voffset
            X_district += hoffset
                
            text = plt.text(np.mean(X_district), np.mean(Y_district), text,
                            color=label_colour,
                          ha='center', va='bottom', size=size)
            if label_shadow:
                text.set_path_effects([path_effects.Stroke(linewidth=2, alpha=.5, foreground=label_shadow),
                       path_effects.Normal()])
                       
            '''           
            plt.annotate(text, 
                         xy=(np.mean(X_district), np.mean(Y_district)),
                         xytext=(np.mean(X_district), np.mean(Y_district)),
                            bbox=dict(boxstyle="square", fc=(1,1,1), 
                                alpha=.7,ec="none"))'''
                                
def circle_district(district, size, color='b'):
    r = sf_districts.records()
    for i_record in range(len(r)):
        if r[i_record][5].lower() == district.lower():
            coords = np.array(district_shapes[i_record].points)
            X_district = np.mean(coords[:,0])
            Y_district = np.mean(coords[:,1])
    try:
        patch = Circle((X_district, Y_district), size, color=color, alpha=.5)
        plt.gca().add_patch(patch)
    except:
        print 'Problem with district "%s"' % (district)
 

def colour_code_districts(values, cmap=plt.cm.winter, scale=None, 
                          showcolorbar=True, outline_colour=(0,0,0)):
    '''Colour code districts according to numerical values. 
    scale: list containing the minimum and maximum values for the colour range.
    cmap: colormap function to be used.
    values: dict mapping district names to numbers in 0-1 range.
    '''
    if not scale:
        scale = [min(values.values()), max(values.values())]
    minval = np.inf
    maxval = 0
    for district in values.keys():
        v = max(scale[0],values[district])
        v = min(v, scale[1])
        if v<minval:
            minval = v
        if v>maxval:
            maxval = v
        plot_single_district(district, fill_colour=cmap(1.0*(v-scale[0])/(scale[1]-scale[0])),
                             outline_colour=outline_colour)
    if showcolorbar:
        plt.imshow([[scale[0],scale[0]],[scale[1],scale[1]]],cmap=cmap)
        plt.colorbar()

def plot_uganda_cropped_image(img, cmap=None):
    '''Plot an image for which the edges correspond to the minimum and 
    maximum longitudes and latitudes of Uganda.'''
    ax = plt.gca()
    im = ax.imshow(img,
                       extent=[UG_min_lon, UG_max_lon,
                               UG_min_lat, UG_max_lat],
                        interpolation=None)
    if cmap:
        im.cmap = cmap
        
def prettify_map(land_resolution=700, 
                 outline_colour = (.5,.5,.5),
                 water_colour=(192,232,255),
                 label_towns=True,
                 show_roads=True,
                 show_reserves=True,
                 show_water=True,
                 label_colour='white',
                 label_shadow='black',
                 road_colour=(.7,.7,.7)):
    '''Colour water bodies.'''
    if show_water:
        global lm
        if lm==None:
            p = water_shapes[0].parts
            p = np.hstack((p,len(water_shapes[0].points)-1))
            land_mask = np.ones((land_resolution, land_resolution))
            for i in range(1,len(p)): 
                water_coords = np.array(water_shapes[0].points[p[i-1]:p[i]])
                if water_coords.shape[0]>100:
                    X_img = (water_coords[:,0] - xlim_l)/((xlim_u-xlim_l)/land_resolution)
                    Y_img = land_resolution - (water_coords[:,1] - ylim_l)/((ylim_u-ylim_l)/land_resolution)
            
                    pts = np.vstack((X_img, Y_img))
                    pts = np.int32(np.round(pts))
            
                    if X_img[0]<land_resolution and Y_img[0]<land_resolution and (land_mask[Y_img[0], X_img[0]]==0):
                        cv2.fillPoly(land_mask, [pts.transpose()], 1)
                    else:
                        cv2.fillPoly(land_mask, [pts.transpose()], 0)
            lm = np.ones((land_resolution,land_resolution,4))
            lm[:,:,0] = water_colour[0]/255.
            lm[:,:,1] = water_colour[1]/255.
            lm[:,:,2] = water_colour[2]/255.
            lm[:,:,3] = 1-land_mask
                      
        plt.imshow(lm, extent=[xlim_l, xlim_u, ylim_l, ylim_u], interpolation='bilinear')
    
    '''Draw roads.'''
    if show_roads:
        for i in range(len(roads_shapes)): 
            road_coords = np.array(roads_shapes[i].points)
            p = roads_shapes[i].parts
            p.append(road_coords.shape[0])
            for part in range(len(p)-1):
                plt.plot(road_coords[p[part]:p[part+1],0], road_coords[p[part]:p[part+1],1],color=road_colour)
    
    '''Draw nature reserves'''
    if show_reserves:
        for i in range(len(reserves_shapes)): 
            reserves_coords = np.array(reserves_shapes[i].points)
            p = reserves_shapes[i].parts
            if p[-1] != reserves_coords.shape[0]:
                p.append(reserves_coords.shape[0])
            for part in range(len(p)-1):
                plt.fill(reserves_coords[p[part]:p[part+1],0], reserves_coords[p[part]:p[part+1],1],color='g',alpha=.05)    
    
    '''Label towns.'''
    if label_towns:
        textmargin = .08
    
        towns = {}
        towns['Kampala'] = [32.5994497,0.3130293]
        towns['Gulu'] = [32.300244,2.786431]
        towns['Mbarara'] = [30.649205, -0.613223]
        towns['Kabale'] = [30.0271899,-1.2464806]
        towns['Fort Portal'] = [30.3095804,0.638414]
        towns['Jinja'] = [33.219841,0.441785]
        towns['Mbale'] = [34.185265, 1.066575]
        towns['Lira'] = [32.901238, 2.256696]
        towns['Arua'] = [30.907219, 3.034516] 
        towns['Mubende'] = [31.395179, 0.556137]
        towns['Kitgum'] = [32.883209, 3.290830] 
    
        for town in towns.keys():
            plt.plot(towns[town][0],towns[town][1],'o', color='white', alpha=.8,     markersize=5)
            text = plt.text(towns[town][0]+textmargin,towns[town][1],town, color=label_colour,
                          ha='left', va='center', size=10)
            text.set_path_effects([path_effects.Stroke(linewidth=2, alpha=.7, foreground=label_shadow),
                       path_effects.Normal()])
                 
    plot_country_outline(outline_colour=outline_colour)
        
    f = plt.gcf()
    f.set_figheight(10)
    f.set_figwidth(10)
    
                        
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    (From matplotlib documentation)
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)

def ndvi_cmap():
    c = matplotlib.colors.ColorConverter().to_rgb
    cmap = make_colormap(
    [c('white'), .35,
     c('white'), c('tan'), .55,
     c('tan'), c('green'), .6,
     c('green'), c('darkgreen'), .65,
     c('darkgreen'), c('black')]) 
    return cmap

                        
if __name__=='__main__':
    #prettify_map(label_colour='k', label_shadow='w')
    '''
    plt.subplot(121)
    img = np.load('data/ndvi/ndvi-2011-01-31.npz')['arr_0']
    plot_uganda_cropped_image(img, cmap=ndvi_cmap())

    '''
    plot_country_outline()
    circle_district('Kampala', 50)
    '''
    plt.subplot(122)
    plot_country_outline()
    
    plot_single_district('mukono', fill_colour=(1,0,0))
    
    v = {'Abim': 0.6808905380333952,
         'Adjumani': 0.6839360222531293,
         'Agago': 0.6440422322775264,
         'Alebtong': 0.643383530919577,
         'Amolatar': 0.6663019693654267,
         'Amudat': 0.6392045454545454,
         'Amuria': 0.6828969904575484,
         'Amuru': 0.636854119111849,
         'Apac': 0.6631173538740371,
         'Arua': 0.67703678239945,
         'Budaka': 0.6903677758318739,
         'Bududa': 0.6507378777231202,
         'Bugiri': 0.6588360918312867,
         'Buhweju': 0.6333333333333333,
         'Buikwe': 0.48772112382934446,
         'Bukedea': 0.6846758349705304,
         'Bukomansimbi': 0.631578947368421,
         'Bukwo': 0.6780141843971631,
         'Bulambuli': 0.6983877310263469,
         'Buliisa': 0.6622833233711894,
         'Bundibugyo': 0.6659869494290375,
         'Bushenyi': 0.654280338664158,
         'Busia': 0.7348466746316209,
         'Butaleja': 0.655064935064935,
         'Butambala': 0.5700280112044818,
         'Buvuma': 0.6239669421487604,
         'Buyende': 0.6573248407643312,
         'Dokolo': 0.6730091613812544,
         'Gomba': 0.6114537444933921,
         'Gulu': 0.6761640798226164,
         'Hoima': 0.6542008196721312,
         'Ibanda': 0.6525630593978845,
         'Iganga': 0.6523304660932187,
         'Isingiro': 0.6469879518072289,
         'Jinja': 0.5888086865321734,
         'Kaabong': 0.6823184152604549,
         'Kabale': 0.6365814696485623,
         'Kabarole': 0.6645893057459803,
         'Kaberamaido': 0.6477891526900458,
         'Kagadi': 0.0,
         'Kalangala': 0.6123595505617978,
         'Kaliro': 0.6099722991689751,
         'Kalungu': 0.5833333333333334,
         'Kampala': 0.6242496192097482,
         'Kamuli': 0.6683760683760683,
         'Kamwenge': 0.6884389288047028,
         'Kanungu': 0.6586457750419697,
         'Kapchorwa': 0.6748729972645565,
         'Kasese': 0.6528465346534653,
         'Katakwi': 0.6283783783783784,
         'Kayunga': 0.6375404530744336,
         'Kibaale': 0.6517857142857143,
         'Kiboga': 0.5955284552845529,
         'Kibuku': 0.6621499548328816,
         'Kiruhura': 0.6581673306772908,
         'Kiryandongo': 0.6394825646794151,
         'Kisoro': 0.6254699248120301,
         'Kitgum': 0.6728562801932367,
         'Koboko': 0.7015663643858203,
         'Kole': 0.6646670665866826,
         'Kotido': 0.6942811330839124,
         'Kumi': 0.6560963618485742,
         'Kween': 0.6879063719115734,
         'Kyankwanzi': 0.6246537396121884,
         'Kyegegwa': 0.6739869281045752,
         'Kyenjojo': 0.6610630407911001,
         'Lamwo': 0.6934326710816777,
         'Lira': 0.6633351180914986,
         'Luuka': 0.6679611650485436,
         'Luwero': 0.6750803120697567,
         'Lwengo': 0.6393337604099936,
         'Lyantonde': 0.6116015132408575,
         'Manafwa': 0.6545848375451263,
         'Maracha': 0.6964928057553957,
         'Masaka': 0.6084129242023979,
         'Masindi': 0.6513687600644122,
         'Mayuge': 0.589406779661017,
         'Mbale': 0.6542984219825748,
         'Mbarara': 0.6451204055766794,
         'Mitooma': 0.6253284287966369,
         'Mityana': 0.6370165745856353,
         'Moroto': 0.657258064516129,
         'Moyo': 0.6862505099959201,
         'Mpigi': 0.6288384512683578,
         'Mubende': 0.6417957635156497,
         'Mukono': 0.6488302061616864,
         'Nakapiripirit': 0.6768826619964974,
         'Nakaseke': 0.706630336058129,
         'Nakasongola': 0.6742902208201893,
         'Namayingo': 0.5794291868605277,
         'Namutumba': 0.6392367322599881,
         'Napak': 0.6633906633906634,
         'Nebbi': 0.663771364527879,
         'Ngora': 0.6777920410783055,
         'Ntoroko': 0.6553030303030303,
         'Ntungamo': 0.6422817330720662,
         'Nwoya': 0.6596413966656182,
         'Otuke': 0.6658943466172382,
         'Oyam': 0.6806247644570997,
         'Pader': 0.6727075588599752,
         'Pallisa': 0.6641944787803873,
         'Rakai': 0.6502930505163271,
         'Rubirizi': 0.7036809815950921,
         'Rukungiri': 0.6842105263157895,
         'Sembabule': 0.6610169491525424,
         'Serere': 0.6514563106796116,
         'Sheema': 0.6443741527338455,
         'Sironko': 0.6712927268054485,
         'Soroti': 0.6413461538461539,
         'Tororo': 0.6609923940601231,
         'Wakiso': 0.6400414937759336,
         'Yumbe': 0.6811298776936517,
         'Zombo': 0.6753650201926064}
 
    colour_code_districts(v, scale=[.5,.8], outline_colour=None)
    
    annotate_district('Arua','Arua')
    '''
