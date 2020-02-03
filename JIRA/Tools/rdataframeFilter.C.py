import ROOT
c  = ROOT.TCanvas("canvas","canvas")
#
ROOT.TGrid.Connect("alien","miranov")

ROOT.gSystem.Exec("alien_find /alice/sim/2019/LHC19f5b_3/296433/SpacePointCalibrationMerge Filter*root  | sed s_/alice_alien:///alice_> filtered_LHC19f5b_3.list")
ROOT.gSystem.Exec("alien_find /alice/sim/2019/LHC19f5b_2/296433/SpacePointCalibrationMerge Filter*root  | sed s_/alice_alien:///alice_> filtered_LHC19f5b_2.list")
ROOT.gSystem.Exec("alien_find /alice/sim/2019/LHC19f5b/296433/SpacePointCalibrationMerge Filter*root  | sed s_/alice_alien:///alice_> filtered_LHC19f5b.list")
ROOT.gSystem.GetFromPipe("wc filtered*.list")

tree = ROOT.AliXRDPROOFtoolkit.MakeChainRandom("filtered_LHC19f5b_3.list","highPt","",20,0)
tree.SetMarkerStyle(21); tree.SetMarkerSize(0.5)
ROOT.AliDrawStyle.SetDefaults()
ROOT.AliDrawStyle.ApplyStyle("figTemplate")
tree.SetCacheSize(200000000)

ROOT.gROOT.LoadMacro("/eos/user/m/mivanov/github/RootInteractiveTest/JIRA/Tools/rdataframeFilter.C")
ROOT.makeRDFrameSnapshot0(tree,"test.root",5)


f = ROOT.TFile.Open("test.root")
