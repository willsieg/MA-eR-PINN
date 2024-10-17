import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# SELECT VEHICLE AND TRIP:
vehicle_selection = '14'
trip_selection = '100'

df = pd.read_parquet("/home/sieglew/data/processed/v_id983V" + vehicle_selection + "_trip" + trip_selection + ".parquet") 

# select certain section if necessary
y1, y2 =3950, 5380

#df = df.iloc[y1:y2]

fig = px.scatter_mapbox(df, 
                        lat="latitude_cval_ippc", 
                        lon="longitude_cval_ippc", 
                        color_continuous_scale=px.colors.cyclical.IceFire, #[(0, 'orange'), (1,'red')],
                        #zoom=8, 
                        #height=400,
                        #width=1000
                        )

fig.add_trace(go.Scattermapbox(
    mode = "markers+text",
    lon = [df.longitude_cval_ippc.iloc[0], df.longitude_cval_ippc.iloc[-1]], lat = [df.latitude_cval_ippc.iloc[0],df.latitude_cval_ippc.iloc[-1]],
    marker=go.scattermapbox.Marker(size=12, color='red'),
    text = ["  >>> START", "  >>> END"],textposition = "middle right",
    textfont = dict(size=18, color="black", family="Open Sans Bold")))

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},autosize=True,showlegend = False)
fig.show(renderer="browser")     #notebook', 'notebook_connected', 'browser', 'firefox', 'chrome'
