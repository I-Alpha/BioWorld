from matplotlib import colors
from matplotlib.pyplot import colormaps, legend, xcorr, ylim
import pandas as pd 
import plotly.graph_objs as go
from math import sin
from math import cos
from math import radians
import plotly.io as pio
import dash
from dash import dcc
from dash import  html
import plotly.graph_objects as go 
pio.renderers.default = "browser"
pd.options.plotting.backend = "plotly"

#--- FUNCTIONS ----------------------------------------------------------------+ 
 
def plot_frame(settings,  entities): 
    fig = go.Figure()    
    color_dict = { "organism":"#121212" , "food":"#AAFF00"}
    df = pd.DataFrame(entities) 
        
    for key in color_dict:
        fig.add_trace(go.Scatter(x = df['x'].where(df["type"]==key), y =  df['y'].where(df["type"]==key),
                            mode='markers', # 'lines' or 'markers',
                            legendgroup="group1",
                            legendgrouptitle_text="Legend",
                            name=key, line_color=color_dict[key])
        )    
    fig.for_each_trace(
            lambda trace: trace.update(marker_symbol="square") if trace.name == "food" else trace.update(marker_symbol="circle"),
        )
    fig.update_xaxes(range=[-2.3, 3.1])
    fig.update_yaxes(range=[-2.3, 2.3])    
    return fig
 

#--- END ----------------------------------------------------------------------+
