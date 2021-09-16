from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd

data = pd.read_csv(r"C:\Users\haimiti.aerfate\Desktop\ProblemSet3\EPO.csv")

#######################################################################
#Figure 1: Average Number of Patents per Million People (1995-2014)
# mean Patent 1995-2014
plotdata1 = go.Choropleth(
        locations = data['LOCATION'],
        z = data['PEPO_APP'],
        text = (data['Country'], data['PEPO_APP']),
        autocolorscale = False,
        colorscale = 'Picnic',
        showscale = True,  
    )

layout = go.Layout(
    #title = "Average Number of Patents per Million People (1995-2014)",
    geo = dict(scope = "europe")
    )

figure1 = go.Figure(data=[plotdata1], layout=layout)
plot(figure1)
figure1.write_image(r"C:\Users\haimiti.aerfate\Desktop\ProblemSet3\fig1.png")


#######################################################################
#functions/loops to save on redundant lines of code
traces = []
for i in data['LOCATION'].unique():
  small_data = data.loc[data['LOCATION']==i, :]
  traces.append(go.Scatter( # initialize scatter object
    x = small_data['Time'], 
    y = small_data['PEPO_APP'], 
    mode="markers+lines", 
    name=' in {}'.format(i)))


#######################################################################
#Figure 2: Distribution of Patents per Million People (1995-2014)
trace1 = go.Box(
  y = small_data['PEPO_APP'],
  name='Mean & SD',
  marker=dict(
    color='rgb(10, 140, 208)',
  ),
  boxmean='sd' # Shows quartiles AND Std Dev on plot
)

#layout=go.Layout(title="Figure 2: Distribution of Patents per Million People (1995-2014)")
                 
plotdata = go.Data([trace1])
figure = go.Figure(data=plotdata, layout=layout)
plot(figure)

#save
figure.write_image(r"C:\Users\haimiti.aerfate\Desktop\ProblemSet3\fig2.png")


#######################################################################
#Figure 3: Patent and Real FDI Stock per Capita (1995-2014)
trace = go.Scatter( # initialize scatter object
   x = small_data['Time'], 
   y = small_data['PEPO_APP'], 
  marker =  {'color': 'green', # choose the marker color
    'symbol': 0, # choose a shape
    #'size': 20
    }, # choose a size
    line=dict(
        shape='spline' # spline smoothing
    ),
    text=['Time: ' + str(i) for i in list(range(1995,2014))], # hover text
    name='Patent',
    connectgaps=True) # name for legends
    
trace2 = go.Scatter( # initialize scatter object
   x = small_data['Time'], 
   y = small_data['PFDI_sto_U'], 
  marker =  {'color': 'blue', # choose the marker color
    'symbol': 0, # choose a shape
    #'size': 20
    }, # choose a size
    line=dict(
        shape='spline' # spline smoothing
    ),
    text=['Time: ' + str(i) for i in list(range(1995,2014))], # hover text
    name='Real FDI Stock per Capita (USD)',
    yaxis='y2', # name for legends
    connectgaps=True) #connect missing observations

plotdata=go.Data([trace, trace2]) # Process the plots

layout=go.Layout(#title="Figure 3: Patent and Real FDI Stock per Capita (1995-2014)", 
                 # configure the plot
  xaxis={'title':'Year',
         'showgrid':False},  # layout and name
  yaxis={'title':'Number of Patents per Million People',
         'showgrid':False},
  yaxis2={'title':"Real FDI Stock  per Capita (Million USD)",
          'overlaying': 'y',
          'side':'right',
          'showgrid':False})  # the axes.

figure=go.Figure(data=plotdata, layout=layout)
# combine data and layout code

plot(figure) # Render the plots

#save
figure.write_image(r"C:\Users\haimiti.aerfate\Desktop\ProblemSet3\fig3.png", width=1500, height=700)


#######################################################################
#Figure 4: Patent and Real FDI Stock (1995-2014)
trace = go.Scatter( # initialize scatter object
   x = small_data['Time'], 
   y = small_data['PEPO_APP'], 
  marker =  {'color': 'green', # choose the marker color
    'symbol': 0, # choose a shape
    #'size': 20
    }, # choose a size
    line=dict(
        shape='spline' # spline smoothing
    ),
    text=['Time: ' + str(i) for i in list(range(1995,2014))], # hover text
    name='Patent',
    connectgaps=True) # name for legends
    
trace2 = go.Scatter( # initialize scatter object
   x = small_data['Time'], 
   y = small_data['FDI_sto_U'], 
  marker =  {'color': 'blue', # choose the marker color
    'symbol': 0, # choose a shape
    #'size': 20
    }, # choose a size
    line=dict(
        shape='spline' # spline smoothing
    ),
    text=['Time: ' + str(i) for i in list(range(1995,2014))], # hover text
    name='Real FDI Stock (Million USD)',
    yaxis='y2', # name for legends 
    connectgaps=True) #connect missing observations

plotdata=go.Data([trace, trace2]) # Process the plots

layout=go.Layout(#title="Figure 4: Patent and Real FDI Stock (1995-2014)", 
                 # configure the plot
  xaxis={'title':'Year',
         'showgrid':False},  # layout and name
  yaxis={'title':'Number of Patents per Million People',
         'showgrid':False},
  yaxis2={'title':"Real FDI Stock (Million USD)",
          'overlaying': 'y',
          'side':'right',
          'showgrid':False})  # the axes.

figure=go.Figure(data=plotdata, layout=layout)
# combine data and layout code

plot(figure) # Render the plots

#save
figure.write_image(r"C:\Users\haimiti.aerfate\Desktop\ProblemSet3\fig4.png", width=1500, height=700)
