# from
from RootInteractive.Tools.aliTreePlayer import *
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from RootInteractive.Tools.aliTreePlayer import *
from bokeh.io import curdoc
import os
import sys
import pytest
from ROOT import TFile, gSystem

output_file("performanceMap.html")
# import logging


treeAlphaV, treeListAlphaV, fileListAlphaV = LoadTrees("cat performance.list",".*_alphaVDist","xxx","",0)
findSelectedBranches(treeAlphaV,[".*Center"],exclude=[".*his.*"])
findSelectedBranches(treeAlphaV,[".*TRD.*ovarP1.*meanG"],exclude=[".*XXX.*"])


dfVariables=tree2Panda(treeAlphaV,[".*Center"],"entries>=0", exclude=[".*his.*"])
dfCovar=tree2Panda(treeAlphaV,[".*TRD.*ovarP1.*meanG"],selection="entries>0",exclude=[".*XXX.*"])

dfCovar=tree2Panda(treeAlphaV,[".*TRD.*hisCovarP.*ITS.*TRD.*_alphaVDist.*meanG",""], "entries>=0",exclude=[".*XXX.*"]),columnMask=[["ITS_TRDv_qPt_tgl_alphaVDist_meanG",""]])



