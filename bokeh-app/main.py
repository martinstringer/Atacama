# To run this code, go to command line (not python shell) and enter
# bokeh serve --show read_paths.py

import numpy as np
import geopandas as gpd
from pandas import read_csv
from bokeh.plotting import figure, output_file, show
from bokeh.models import Range1d,ColumnDataSource,TapTool,LabelSet
from bokeh.tile_providers import get_provider, STAMEN_TERRAIN
from bokeh.colors import RGB
from bokeh.layouts import column,row,gridplot,widgetbox
from bokeh.io import curdoc
from bokeh.transform import linear_cmap
from matplotlib import cm
from bokeh.events import Tap,DoubleTap,MouseMove
from bokeh.models.widgets import DataTable,TableColumn

data_folder = 'data/'

# Read in the shapefile with the hexagon data
hexagons = gpd.read_file(data_folder+'Topo_hex.shp')
# Read average heights from csv file
data = read_csv(data_folder+'hex_centroid_cost.csv')
# Merge two data sets and convert to desired coordinate system
hexagons = hexagons.merge(data,on='GRID_ID').to_crs("Web Mercator")

# Pull out the vertices of all the hexagons as a list
vertices = [g for g in hexagons.geometry]
Nh = len(vertices)
# Set up lists to contain x and y coordinates of vertices
xh = []
yh = []
x=np.zeros(Nh)
y=np.zeros(Nh)
z=np.zeros(Nh)
# Populate lists from shapefile data
for j in range(Nh):
    xv,yv = np.array(vertices[j].exterior.coords.xy)
    xh.append(xv)
    yh.append(yv)
    xy = np.array(vertices[j].centroid.xy)
    x[j] = xy[0][0]
    y[j] = xy[1][0]         
    #z[j] = hexagons.TOPO_mean[j]
    z[j] = hexagons.COMBINED_mean[j]
    
# Use the range of the vertices to set bounds of map
xmin = np.min(xh)
xmax=np.max(xh)
dx = xmax-xmin
ymin = np.min(yh)
ymax= np.max(yh)
xmin = xmin-0.1*dx
xmax=xmax+0.1*dx
h = 700
w = int(round(h*(xmax-xmin)/(ymax-ymin),0))

# Choose default alpha value for hexagon shading
al = 0.8

# Read the array with all the optimised paths
paths = np.load(data_folder+'paths_array.npy',allow_pickle=True)
Np = paths.shape[0]

# Read the array with cost of each paths
path_costs = np.load(data_folder+'path_costs_array.npy',allow_pickle=True)/1e5 

# Create data source containing grid data
grid_source=ColumnDataSource(dict(x=x,y=y,z=z,xh=xh,yh=yh,al=np.ones(Nh)*al))

# Set up map frame
p = figure(x_axis_type="mercator", y_axis_type="mercator",
           tools= "pan,wheel_zoom,tap",active_scroll="wheel_zoom",
           x_range=Range1d(xmin,xmax), y_range=Range1d(ymin,ymax),
           height=h,width=w,
           title='Click to create demand point.          Double-click to create source')
p.xgrid.visible = False
p.ygrid.visible = False
p.axis.visible = False

# Add background map
background_map=True
if background_map:
    tile_provider = get_provider(STAMEN_TERRAIN)
    p.add_tile(tile_provider)

# Create cost pallete
cost_rgb = (255 * cm.RdYlGn(range(256))).astype('int')[::-1]
cost_palette = [RGB(*tuple(rgb)).to_hex() for rgb in cost_rgb]
# Create color scheme for hexagon grid using palette
mapper = linear_cmap(field_name='z', palette=cost_palette,low=0,high=np.max(z))

# Set up data source for source nodes
water_source = ColumnDataSource(dict(xs=[],ys=[],xc=[],yc=[],ns=[]))

# Set up data source for demand nodes 
water_demand = ColumnDataSource(dict(xd=[],yd=[],nd=[],node=[]))

# Set up data source for network 
network = ColumnDataSource(dict(xn=[],yn=[],nn=[],cn=[],cmax=[],link=[],ref=[]))

# Set up data source for new paths
new_paths_data = ColumnDataSource(dict(xo=[],yo=[],co=[]))

optimal_path_data = ColumnDataSource(dict(xp=[],yp=[],nopt=[],copt=[]))

# Add hexagon grid
h = p.patches('xh', 'yh', alpha='al',selection_alpha='al',nonselection_alpha='al',source=grid_source,
                  color=mapper,line_color='white',line_alpha='al',
                  line_width=2,hover_line_color='cyan')

# Set up network lines
network_lines = p.multi_line('xn','yn',source=network,line_color='blue',line_width=4,alpha=al,
                             selection_line_alpha=al,nonselection_line_alpha=al,line_alpha=al)
# Set up highlighting of possible connection paths
new_path_lines = p.multi_line('xo','yo',source=new_paths_data,line_color='cyan',line_width=3,alpha=1,
                              selection_line_alpha=1,nonselection_line_alpha=1,line_alpha=1)
# Set up highlighting of optimal path to existing network
optimal_path_line = p.multi_line('xp','yp',source=optimal_path_data,line_color='blue',line_dash="dashed",line_width=3,
                           selection_line_alpha=1,nonselection_line_alpha=1,line_alpha=1)
# Set up sources points
sources = p.patches('xs','ys',hatch_color='cyan',hatch_pattern='horizontal_wave',
                    line_color='cyan',line_width=0.5,selection_line_alpha=al,nonselection_line_alpha=al,
                    fill_color='blue',fill_alpha=al,nonselection_fill_alpha=al,selection_fill_alpha=al,
                    alpha=al,nonselection_alpha=al,selection_alpha=al,source=water_source)
# Set up demand points
demand = p.circle('xd','yd',source=water_demand,fill_color='blue',alpha=1,size=20,line_color=None,
                  fill_alpha=al,nonselection_fill_alpha=al,selection_fill_alpha=al)
# And numeric lables on demand points
labels = LabelSet(x='xd',y='yd',text='node',text_color='white',text_align='center',text_baseline='middle',
                  x_offset=0,y_offset=0,source=water_demand,render_mode='canvas')
p.add_layout(labels)

# Set up list for all cells in network
network_nodes = np.array([],dtype=int)

i_hover=Nh+2
# Create rule for showing paths on hover
def show_paths(event):
    global i_hover,paths,x,y,network_nodes
    d2 = (x-event.x)**2 + (y-event.y)**2
    i = np.argmin(d2)
    # Check if mouse cursor is outside hexagon grid or on existing network
    if (d2[i]>4e8) or (i in network_nodes):
        # If mouse cursor has left hexagon grid, remove path highlighting
        new_path_options = ColumnDataSource(dict(xo=[],yo=[],co=[]))
        new_paths_data.data.update(new_path_options.data)
        optimal_path = ColumnDataSource(dict(xp=[],yp=[],nopt=[],copt=[]))
        optimal_path_data.data.update(optimal_path.data)
        i_hover=i
    else:
        # Check if mouse has moved to a new cell
        if (i!=i_hover) and np.logical_not:
            #print('Mouse moved to cell',i,(x[i]-x[i_hover])**2 + (y[i]-y[i_hover])**2)
            N_nodes = network_nodes.size
            # Check if there is any network to show connections to
            if N_nodes>0:
                # If so, find shortest path to the existing network
                path_options = paths[i,network_nodes]
                path_option_costs = path_costs[i,network_nodes]
                k = np.argmin(path_option_costs)
                optimal_route = path_options[k]
                optimal_path = ColumnDataSource(dict(xp=[x[optimal_route]],yp=[y[optimal_route]],
                                                     nopt=[optimal_route],copt=[path_option_costs[k]]))
                optimal_path_data.data.update(optimal_path.data)
                source_cells = water_source.data['ns']
                if len(source_cells)>0:
                    xo = []
                    yo = []
                    co = []
                    for j in source_cells:
                        route = paths[i,j]
                        xo.append(x[route])
                        yo.append(y[route])
                        co.append(path_costs[i,j])
                    # Update direct-to-source routes
                    new_path_options = ColumnDataSource(dict(xo=xo,yo=yo,co=co))
                else:
                    # No sources yet, so just use path to network
                    new_path_options = ColumnDataSource(dict(xo=[x[optimal_route]],yo=[y[optimal_route]],
                                                             co=[path_option_costs[k]]))
                new_paths_data.data.update(new_path_options.data)     
            i_hover=i
        
# Create rule for updating paths on click
def create_demand_point(event):
    global paths,path_costs,x,y,network_nodes
    i = grid_source.selected.indices[0]
    # Check that we don't already have this point in the network
    if (i in network_nodes):
        print('Node is already in network')
    else:
        print('New demand point added at cell:',i)
        # If new point, add this cell to list of demand nodes
        xd = water_demand.data['xd']
        xd.append(x[i])
        yd = water_demand.data['yd']
        yd.append(y[i])
        nd = water_demand.data['nd']
        nd.append(i)
        node = water_demand.data['node']
        if len(node)>0:
            node.append(str(int(node[-1])+1))
        else:
            node.append('1')
        new_demand = ColumnDataSource(dict(xd=xd,yd=yd,nd=nd,node=node))
        water_demand.data.update(new_demand.data)
        N_nodes = network_nodes.size
        #print('Existing network nodes:',network_nodes)
        # Check if there is any network to connect to
        if N_nodes>0:
            # Connect new node to existing network
            route = optimal_path_data.data['nopt']
            new_segment = {
                'nn' : [route],
                'xn' : [x[route]],
                'yn' : [y[route]],
                'cn' : optimal_path_data.data['copt'],
                'cmax' : [np.max(new_paths_data.data['co'])],
                'link' : [len(network.data['link'])+1],
                'ref' : [node[-1]]
            }
            network.stream(new_segment)
            print('Added point:',network.data['link'],network.data['cn'])
            # Add cells on route to network nodes
            network_nodes = np.unique(np.append(network_nodes,route))
        else:
            # For the first demand point, just update list of network nodes
            network_nodes = np.append(network_nodes,i)
        print('Updated network nodes:',network_nodes)
            
def create_source(event):
    global x,y,network_nodes,xh,yh
    #print('Double Tap registered',event.x,event.y)
    d2 = (x-event.x)**2 + (y-event.y)**2
    i = np.argmin(d2)
    # Check if mouse cursor is outside hexagon grid or on existing network
    if (d2[i]>4e8):
        print('Cursor is outside grid area')
    # Check that we don't already have this point in the network
    elif i in network_nodes:
        print('Node is already in network')
    else:
        print('New water source added at cell:',i)
        #print('Existing network nodes:',network_nodes)
        # Check to see if there is any network yet
        if network_nodes.size>0:
            # If there is, connect new source to network
            route = optimal_path_data.data['nopt']
            new_segment = {
                'nn' : [route],
                'xn' : [x[route]],
                'yn' : [y[route]],
                'cn' : optimal_path_data.data['copt'],
                'cmax' : [np.min(new_paths_data.data['co'])],
                'link' : [len(network.data['link'])+1],
                'ref' : ['S'+str(len(water_source.data['xs'])+1)]
            }
            network.stream(new_segment)
            # Add cells on route to network nodes
            network_nodes = np.unique(np.append(network_nodes,route))
            #if len(water_source.data['ns'])==0:
                # If this is the first source, need to update network link costs.   
        # Add this cell to list of sources
        xs = water_source.data['xs']
        xs.append(xh[i])
        ys = water_source.data['ys']
        ys.append(yh[i])
        xc = water_source.data['xc']
        xc.append(x[i])
        yc = water_source.data['yc']
        yc.append(y[i])
        ns = water_source.data['ns']
        ns.append(i)
        new_source = ColumnDataSource(dict(xs=xs,ys=ys,xc=xc,yc=yc,ns=ns))
        water_source.data.update(new_source.data)
        # Add this cell to list of network nodes
        network_nodes=np.append(network_nodes,i)
        print('Updated network nodes:',network_nodes)         
    
taptool = p.select(type=TapTool,renderers=[h])
p.on_event(Tap,create_demand_point)
doubletaptool = p.select(type=DoubleTap,renderers=[h])
p.on_event(DoubleTap,create_source)
hovertool = p.select(type=MouseMove,renderers=[h])
p.on_event(MouseMove,show_paths)

# Add table
columns = [
        TableColumn(field="ref", title="Link"),
        TableColumn(field="cn", title="Cost if network shared"),
        TableColumn(field="cmax", title="Cost to connect directly to source")
    ]
q = DataTable(source=network, columns=columns, width=w, height=250,index_position=None)


# Add graph to illustrate
r = figure(width=w, height=250,tools="")#,y_range=Range1d(0,1))
r.xaxis.major_tick_line_color = None
r.xaxis.minor_tick_line_color = None
r.xaxis.major_label_text_color = None
r.circle(2,1,alpha=0)
r.vbar(x='link',top='cmax',width=0.4,source=network,color='cyan',alpha=0.5*al,line_color=None)
r.vbar(x='link',top='cn',width=0.4,source=network,color='blue',alpha=al,line_color=None)
#r.circle('link','cn',source=network,fill_color='blue',alpha=al,size=20,line_color=None)
bar_labels = LabelSet(x='link',y='cn',text='ref',text_color='white',text_align='center',text_baseline='top',
                       x_offset=0,y_offset=-10,source=network,render_mode='canvas')
r.add_layout(bar_labels)
r.xaxis.axis_label = 'Network segment'
r.yaxis.axis_label = 'Cost'# (relative)'

page = gridplot([[p,column(widgetbox(q),r)],[]],toolbar_location="left")

curdoc().add_root(page)
   
