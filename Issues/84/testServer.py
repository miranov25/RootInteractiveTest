from random import random
from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
# RootInteracive
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
import os
import sys
import pytest

counter=0

df = pd.DataFrame(np.random.random_sample(size=(20000, 6)), columns=list('ABCDEF'))
initMetadata(df)
df.eval("Bool=A>0.5", inplace=True)
df["AA"]=(df.A*10).round(0)/10.
df["CC"]=(df.C*5).round(0)
df["DD"]=(df.D*2).round(0)
df["EE"]=(df.E*4).round(0)
df['errY']=df.A*0.02+0.02;
df.head(10)
df.meta.metaData = {'A.AxisTitle': "A (cm)", 'B.AxisTitle': "B (cm/s)", 'C.AxisTitle': "C (s)", 'D.AxisTitle': "D (a.u.)", 'Bool.AxisTitle': "A>half", 'E.AxisTitle': "Category"}

figureArray = [
#   ['A'], ['C-A'], {"color": "red", "size": 7, "colorZvar":"C", "filter": "A<0.5"}],
    [['A'], ['A*A-C*C'], {"color": "red", "size": 2, "colorZvar": "A",  "errY": "errY", "errX":"0.01"}],
    [['A'], ['C+A', 'C-A', 'A/A']],
    [['B'], ['C+B', 'C-B'], {"color": "red", "size": 7, "colorZvar": "C", "errY": "errY", "rescaleColorMapper": True }],
    [['D'], ['(A+B+C)*D'], {"size": 10, "errY": "errY"} ],
    [['D'], ['D*10'], {"size": 10, "errY": "errY"}],
]
tooltips = [("VarA", "(@A)"), ("VarB", "(@B)"), ("VarC", "(@C)"), ("VarD", "(@D)")]
widgetParams=[
    ['range', ['A']],
    ['range', ['B', 0, 1, 0.1, 0, 1]],
    ['range', ['C'], {'type': 'minmax'}],
    ['range', ['D'], {'type': 'sigma', 'bins': 10, 'sigma': 3}],
    ['range', ['E'], {'type': 'sigmaMed', 'bins': 10, 'sigma': 3}],
    ['slider', ['AA'], {'bins': 10}],
    ['multiSelect', ["DD", 0, 1, 2, 3]],
    ['select',["CC", 0, 1, 2, 3]],
    #['slider','F', ['@min()','@max()','@med','@min()','@median()+3*#tlm()']], # to be implmneted
]
widgetLayoutDesc=[[0, 1, 2], [3, 4, 5], [6, 7], {'sizing_mode': 'scale_width'}]

figureLayoutDesc=[
    [0, 1, 2, {'commonX': 1, 'y_visible': 1, 'x_visible':1, 'plot_height': 300}],
    [3, 4, {'plot_height': 100, 'x_visible': 1, 'y_visible': 2}],
    {'plot_height': 100, 'sizing_mode': 'scale_width', 'y_visible' : 2}
]
def callback(attr, old, new):
    global i
    print(i)
    i=i+1



myDashboard=bokehDrawSA.fromArray(df, "A>0", figureArray, widgetParams, layout=figureLayoutDesc, tooltips=tooltips, widgetLayout=widgetLayoutDesc, nPointRender=200)
for widget in myDashboard.widgetList.children[0].children:
    widget.on_change('value',callback)
# add a button widget and configure with the call back

def callbackB():
    global counter
    global myDashboard
    cds=myDashboard.cdsSel
    print("callback button",counter)
    counter=counter+1

button = Button(label="Refresh")
button.on_click(callbackB)

# put the button and plot in a layout and add to the document
curdoc().add_root(column(myDashboard.pAll,button))