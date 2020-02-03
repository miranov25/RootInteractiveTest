/*
  .L ~/github/RootInteractiveTest/JIRA/Tools/rdataframeFilter.C

 */
using RNode = ROOT::RDF::RNode;

/// append append Ncr and NCf  position  information
/// \param df - input data frame
/// \return   - output data frame
RNode DFDefineTPCNcrNcfPos(RNode df, std::vector<std::string> &columns){
  std::string  name="";
  for (Int_t row0 = 0; row0 < 80; row0 += 15) {
    Int_t row1=row0+25;
    Float_t R = 83+(row0+12.)*0.75;
    name = Form("NCr%d", row0);
    df = df.Define(name, [row0,row1](AliESDtrack &p) {return p.GetTPCClusterInfo(3, 1, row0, row1);},{"esdTrack."});
    columns.push_back(name);
    name = Form("NCf%d", row0);
    df = df.Define(name, [row0,row1](AliESDtrack &p) {return p.GetTPCClusterInfo(3, 0, row0, row1);},{"esdTrack."});
    columns.push_back(name);
    name = Form("LocalPhi%d", row0);
    df = df.Define(name, [R](AliESDtrack &p, Float_t bz) {return p.GetParameterAtRadius(R, bz, 10);},{"esdTrack.","Bz"});
    columns.push_back(name);
    name = Form("LocalSector%d", row0);
    df = df.Define(name, [R](AliESDtrack &p, Float_t bz) {return p.GetParameterAtRadius(R, bz, 13);},{"esdTrack.","Bz"});
    columns.push_back(name);
  }
  return df;
}

void makeRDFrameSnapshot0(TTree *tree, const char *outputFile, Float_t bz) { //const char * filter, Long64_t nEvents, Double_t dcaCut) {
  // tree = (TTree *) AliXRDPROOFtoolkit::MakeChain("filtered.list", "highPt", 0, 1000000000, 0, 0);
  // treeEvent = (TTree *) AliXRDPROOFtoolkit::MakeChain("filtered.list", "events", 0, 1000000000, 0, 0);
  //
 auto  bunchCrossing =[](ULong64_t gid) {return gid&0xFFF;};
  //
  ROOT::RDataFrame *pdf = new ROOT::RDataFrame(*tree);
  RNode df = pdf->Filter("abs(esdTrack.fD)<4&&abs(esdTrack.fZ)<3");
  //
  df = df.Define("bunchCrossing",bunchCrossing,{"gid"});
  df = df.Define("qP",[](AliESDtrack &p) { return p.GetSign()/p.P();},{"esdTrack."});
  df = df.Define("qPt",[](AliESDtrack &p) { return p.GetSign()/p.Pt();},{"esdTrack."});
  df = df.Define("P",[](AliESDtrack &p) { return p.P();},{"esdTrack."});
  df = df.Define("Ptpc",[](AliESDtrack &p) { return p.GetInnerParam()->P();},{"esdTrack."});
  df = df.Define("tgl",[](AliESDtrack &p) { return p.GetTgl();},{"esdTrack."});

  //
  std::vector <std::string> columns = {"gid", "bunchCrossing",
                                       "qP","P","Ptpc","tgl","qPt",
                                       "cr0_25","cr20_45", "cr40_65","cr60_85",
                                       "cf0_25","cf20_45", "cf40_65","cf60_85",
                                       "pos0_25","pos20_45", "pos40_65","pos60_85"};
  df= DFDefineTPCNcrNcfPos(df,columns);
  df.Snapshot("tree", outputFile, columns);
}

void testDataFrame1(){
  ROOT::RDataFrame d(100); // a RDF that will generate 100 entries (currently empty)
  int x = -1;
  auto dcol = d.Define("x", [&x] { return ++x; }).Define("xx", [&x] { return x*x; });
  dcol.Snapshot("xxx","xxx.root");
}

void testDataFrame2(){
  ROOT::RDataFrame d(100); // a RDF that will generate 100 entries (currently empty)
  int x = -1;
  RNode  dcol = (RNode) (d.Define("x", [&x] { return ++x; }).Define("xx", [&x] { return x*x; }));
  dcol.Snapshot("xxx","xxx.root");
}
