/*
Class to generate benchmark test split accuracies for UCR/UEA benchmark 
* Time Series Classification problems
* 
* oneNearestNeighbour:  These results should tally with those on the UCR website
    ED
    DTW
    DTWCV
* kNearestNeighbour: compare 1NN to kNN with k set through cross validation


*/

package development;


import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import statistics.simulators.DataSimulator;
import statistics.simulators.PolynomialModel;
import statistics.simulators.ShapeletModel;
import weka.attributeSelection.*;
import weka.classifiers.Classifier;
import utilities.ClassifierTools;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.core.elastic_distance_measures.DTW;
import weka.filters.Filter;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.SummaryStats;
import weka.filters.unsupervised.attribute.RemoveUseless;

/**
 *
 * @author ajb
 */
public class NearestNeighbourTSC {
 
    static String[] files=DataSets.fileNames;
    static String path=DataSets.clusterPath;
    private static class Results{
        String name;
        double ed;
        double dtw;
        double dtwcv;
        Results(String s, double x, double y, double z){
            name=s;
            ed=x;
            dtw=y;
            dtwcv=z;
        }
    }
 //UCR Reported Errors for 1-NN, DTW full DTW with window set through CV
    static private Results[] ucrError=initializeUCRResults();
    private static Results[] initializeUCRResults(){
        String[] names=DataSets.uciFileNames;
        double[] ed={0.389,0.467,0.267,0.148,0.35,0.103,0.25,0.426,0.356,0.38,0.065,0.12,0.203,0.286,0.216,0.231,0.369,0.217,0.087,0.63,0.658,0.045,0.246,0.425,0.086,0.316,0.121,0.171,0.12,0.133,0.483,0.038,0.305,0.141,0.151,0.213,0.1,0.12,0.24,0.253,0.09,0.261,0.338,0.35,0.005,0.382,0.17};
        double[] dtw={0.396,0.5,0.267,0.003,0.352,0.349,0.179,0.223,0.208,0.208,0.033,0.23,0.232,0.192,0.17,0.0951,0.31,0.167,0.093,0.623,0.616,0.05,0.131,0.274,0.066,0.263,0.165,0.209,0.135,0.133,0.409,0,0.275,0.169,0.093,0.21,0.05,0.007,0,0.096,0,0.273,0.366,0.342,0.02,0.351,0.164};
        double[] dtwcv={0.391,0.467,0.233,0.004,0.35,0.07,0.179,0.236,0.197,0.18,0.065,0.203,0.192,0.114,0.088,0.242,0.16,0.087,0.588,0.613,0.045,0.131,0.288,0.086,0.253,0.134,0.185,0.129,0.167,0.384,0,0.305,0.141,0.095,0.157,0.062,0.017,0.01,0.132,0.0015,0.227,0.301,0.322,0.005,0.252,0.155,0.164};
        Results[] res= new Results[names.length];
        for(int i=0;i<names.length;i++)
            res[i]=new Results(names[i],ed[i],dtw[i],dtwcv[i]);
        return res;
    }
    public static void recreateUCRTrainTest(OutFile results, boolean normalise) throws Exception{
        Results[] res= initializeUCRResults();
            results.writeLine("fileName,ucrED,ucrDTW,ucrDTWCV,ueaED,ucrDTW,ucrDTWCV");
        for(int i=0;i<DataSets.ucrNames.length;i++){
            String fileName=DataSets.ucrNames[i];
            Instances test=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TEST");
            Instances train=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TRAIN");
            results.writeString(fileName+","+res[i].ed+","+res[i].dtw+","+res[i].dtwcv+",");
            if(normalise){
                NormalizeCase nc = new NormalizeCase();
                train=nc.process(train);
                test=nc.process(test);
            }
    // 1-NN ED
            kNN ed1NN= new kNN(1);
            ed1NN.setDistanceFunction(new EuclideanDistance());
            ed1NN.normalise(false);
            ed1NN.setCrossValidate(false);
            double acc=ClassifierTools.singleTrainTestSplitAccuracy(ed1NN, train, test);
            results.writeString((1-acc)+",");
    // 1-NN DTW My Version
            DTW_1NN dtwMine=new DTW_1NN();
            dtwMine.optimiseWindow(false);
            acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwMine, train, test);
            results.writeString((1-acc)+",");
                results.writeString((1-acc)+",");
    //1-NN DTWC        
            DTW_1NN dtwcv=new DTW_1NN();
            dtwcv.optimiseWindow(true);
            acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwcv, train, test);
            results.writeString((1-acc)+","+dtwcv.getMaxR());
            results.writeString("\n");
        }
    }
    
    
    public static void nearestNeighbourClassifiersTrainTest(String fileName, OutFile results, boolean normalise) throws Exception{
        Instances test=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TEST");
        Instances train=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TRAIN");
        results.writeString(fileName+",");
        if(normalise){
            NormalizeCase nc = new NormalizeCase();
            train=nc.process(train);
            test=nc.process(test);
        }
        
// 1-NN ED
        kNN ed1NN= new kNN(1);
        ed1NN.setDistanceFunction(new EuclideanDistance());
        ed1NN.normalise(false);
        ed1NN.setCrossValidate(false);
        double acc=ClassifierTools.singleTrainTestSplitAccuracy(ed1NN, train, test);
        results.writeString((1-acc)+",");
// k-NN ED
        kNN edkNN= new kNN(100);
        edkNN.setDistanceFunction(new EuclideanDistance());
        edkNN.normalise(false);
        edkNN.setCrossValidate(true);
        acc=ClassifierTools.singleTrainTestSplitAccuracy(edkNN, train, test);
        results.writeString((1-acc)+",");
// 1-NN DTW
        kNN dtw=new kNN(1);
        dtw.setDistanceFunction(new DTW());
        dtw.normalise(false);
        dtw.setCrossValidate(false);
        acc=ClassifierTools.singleTrainTestSplitAccuracy(dtw, train, test);
        results.writeString((1-acc)+",");
// 1-NN DTW My Version
        DTW_1NN dtwMine=new DTW_1NN();
        dtwMine.optimiseWindow(false);
        acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwMine, train, test);
        results.writeString((1-acc)+",");
// k-NN DTW
        kNN dtwK=new kNN(100);
        dtwK.setDistanceFunction(new DTW());
        dtwK.normalise(false);
        dtwK.setCrossValidate(true);
        acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwK, train, test);
        results.writeString((1-acc)+",");
//1-NN DTWCV        
        DTW_1NN dtwcv=new DTW_1NN();
        dtwcv.optimiseWindow(true);
        acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwcv, train, test);
        results.writeString((1-acc)+","+dtwcv.getMaxR());
        results.writeString("\n");
    }
     
    public static void trainTestAll(String fileName, OutFile results, boolean normalise) throws Exception{
        for(String s:DataSets.fileNames)
            nearestNeighbourClassifiersTrainTest(s,results,normalise);
    }
        
    
    public static void nearestNeighbourClassifiersResample(String fileName, OutFile results, boolean normalise, int resamples) throws Exception{
        Instances test=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TEST");
        Instances train=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TRAIN");
        results.writeString(fileName+",");
        if(normalise){
            NormalizeCase nc = new NormalizeCase();
            train=nc.process(train);
            test=nc.process(test);
        }
        Instances all=new Instances(train);
        for(Instance ins:test)
            all.add(ins);
        int testSize=test.numInstances();
        
        for(int i=0;i<resamples;i++){
    //Form new Train/Test split
            all.randomize(new Random());
            Instances tr = new Instances(all);
            Instances te= new Instances(all,0);
            for(int j=0;j<testSize;j++)
                te.add(tr.remove(0));            
            
    // 1-NN ED
            kNN ed1NN= new kNN(1);
            ed1NN.setDistanceFunction(new EuclideanDistance());
            ed1NN.normalise(false);
            ed1NN.setCrossValidate(false);
            double acc=ClassifierTools.singleTrainTestSplitAccuracy(ed1NN, tr, te);
            results.writeString((1-acc)+",");
    // 1-NN DTW My Version
            DTW_1NN dtwMine=new DTW_1NN();
            dtwMine.optimiseWindow(false);
            acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwMine, tr, te);
            results.writeString((1-acc)+",");
    //1-NN DTWC        
            DTW_1NN dtwcv=new DTW_1NN();
            dtwcv.optimiseWindow(true);
            acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwcv, tr, te);
            results.writeString((1-acc)+","+dtwcv.getMaxR());
            results.writeString("\n");
        }
    }
 
    public static void nearestNeighbourClassifiersSingleSample(String fileName, OutFile results, boolean normalise) throws Exception{
        Instances test=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TEST");
        Instances train=utilities.ClassifierTools.loadData(path+fileName+"/"+fileName+"_TRAIN");
        results.writeString(fileName+",");
        if(normalise){
            NormalizeCase nc = new NormalizeCase();
            train=nc.process(train);
            test=nc.process(test);
        }
        Instances all=new Instances(train);
        for(Instance ins:test)
            all.add(ins);
        int testSize=test.numInstances();
    //Form new Train/Test split
        all.randomize(new Random());
        Instances tr = new Instances(all);
        Instances te= new Instances(all,0);
        for(int j=0;j<testSize;j++)
            te.add(tr.remove(0));            
// 1-NN ED
        kNN ed1NN= new kNN(1);
        ed1NN.setDistanceFunction(new EuclideanDistance());
        ed1NN.normalise(false);
        ed1NN.setCrossValidate(false);
        double acc=ClassifierTools.singleTrainTestSplitAccuracy(ed1NN, tr, te);
        results.writeString((1-acc)+",");
// 1-NN DTW My Version
        DTW_1NN dtwMine=new DTW_1NN();
        dtwMine.optimiseWindow(false);
        acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwMine, tr, te);
        results.writeString((1-acc)+",");
//1-NN DTWC        
        DTW_1NN dtwcv=new DTW_1NN();
        dtwcv.optimiseWindow(true);
        acc=ClassifierTools.singleTrainTestSplitAccuracy(dtwcv, tr, te);
        results.writeString((1-acc)+","+dtwcv.getMaxR());
        results.writeString("\n");
    }
 
 
    
    public static void combineTestTrain(){
        OutFile of2 = new OutFile(path+"Missing.csv");
        OutFile of = new OutFile(path+"Summary.csv");
        for(String s:DataSets.fileNames){
            File f=new File(path+s+".csv");
            if(f.exists()){
                InFile in= new InFile(path+s+".csv");
                String str=in.readLine();
                of.writeLine(str);
            }
            else{
                of2.writeLine("\""+s+"\"");
            }
        }
    }    
    public static void combineResample(){
        OutFile of2 = new OutFile(path+"Missing.csv");
        OutFile of = new OutFile(path+"Collated.csv");
        OutFile of3 = new OutFile(path+"MeanStdDev.csv");
        for(String s:DataSets.fileNames){
            File f=new File(path+s+".csv");
            if(f.exists()){
                System.out.println(s+" ");
                InFile in= new InFile(path+s+".csv");
                int reps=in.countLines();
                System.out.println(" reps ="+reps);
                in= new InFile(path+s+".csv");
                String name=in.readString();
                if(!s.equals(name)){
                    System.out.println(" ERROR: Name mismatch s="+s+" from file ="+name);
                }
                double[][] all=new double[3][reps];
                double[] sum=new double[3];
                double[] sumSq=new double[3];
                double r=0;
                if(reps==100){
                    for(int i=0;i<reps;i++){
                        double x=in.readDouble();
                        all[0][i]=x;
                        sum[0]+=x;
                        sumSq[0]+=x*x;
                        x=in.readDouble();
                        all[1][i]=x;
                        sum[1]+=x;
                        sumSq[1]+=x*x;
                        x=in.readDouble();
                        all[2][i]=x;
                        sum[2]+=x;
                        sumSq[2]+=x*x;
                        r+=in.readDouble();
                    }
                    of.writeString(s+","+reps+",");
                    for(int i=0;i<reps;i++)
                        of.writeString(all[0][i]+",");
                    of.writeString(",");
                    for(int i=0;i<reps;i++)
                        of.writeString(all[1][i]+",");
                    of.writeString(",");
                    for(int i=0;i<reps;i++)
                        of.writeString(all[2][i]+",");

                    of.writeString("\n");
                    for(int i=0;i<sum.length;i++)
                        of3.writeString((sum[i]/reps)+",");
              /*      for(int i=0;i<sumSq.length;i++){
                        sumSq[i]=(sumSq[i]*sumSq[i])/(reps-1)-(2*sum[i]*sum[i])/((reps-1)*reps)+(sum[i]*sum[i])/((reps-1)*reps*reps);
//                        sumSq[i]/=reps;
                        sumSq[i]=Math.sqrt(sumSq[i]);
                    }
              */      for(int i=0;i<sumSq.length;i++)
                        of3.writeString((sumSq[i])+",");
                    of3.writeString(","+r/(double)reps);
                    of3.writeString("\n");
                }
                else{
                   of2.writeLine(s+","+reps);     
                }
                    
            }
            else{
                of2.writeLine("\""+s+"\"");
            }
        }
    }    
    public static void combineSingle(String s){
        OutFile of = new OutFile(path+s+".csv");
        of.writeString(s+",");
        for(int i=0;i<100;i++){
            File f=new File(path+s+"\\"+s+i+".csv");
            if(f.exists()){
                InFile in= new InFile(path+s+"\\"+s+i+".csv");
                String str=in.readString();
                str=in.readLine();
                of.writeLine(str);
            }
            else{
                System.out.println(" Error "+i+" does not exist on path  "+path+s+"\\"+s+i+".csv");
                System.exit(0);
            }
        }
    }    
    
    public static void main(String[] args) {
        path=DataSets.dropboxPath;
        path="C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NN Benchmarks\\Resample\\";

        combineResample();
        System.exit(0);
//        trainTestSummary();
        
/*        path="C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NN Benchmarks\\ResampleNorm\\";
        System.exit(0);
        combineSingle("ElectricDevices");
  */ 
        int rep=Integer.parseInt(args[0])-1;
        path=DataSets.clusterPath;
        String problem="UWaveGestureLibraryAll";
        if(args[1]!=null)
            problem=args[1];
        boolean normalise=false; 
        if(args[2]!=null){
            if(Integer.parseInt(args[2])==1)//Normalise
                normalise=true;
            System.out.println(args[0]+"   "+args[1]+"   "+args[2]+" "+Integer.parseInt(args[2]));
        }
            
        OutFile of;
        if(!normalise)
             of= new OutFile("Results/"+problem+rep+".csv");
        else
            of = new OutFile("Results/"+problem+"Norm"+rep+".csv");
        try{
            nearestNeighbourClassifiersSingleSample(problem,of,normalise);
        }catch(Exception e){
            System.out.println("CRASH "+ args[0]+"   "+args[1]+"   "+args[2]+" "+e);
            
        }


//        path="C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NN Benchmarks\\trainTest\\";
//        path=DataSets.dropboxPath;
 //       combineTestTrain();
 //       path="C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NN Benchmarks\\trainTestNorm\\";
 //       combineTestTrain();
//    path="C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NN Benchmarks\\Resample\\";
 //       combineResample();
//        path="C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NN Benchmarks\\ResampleNorm\\";
//        combineResample();
//        OutFile of = new OutFile(resultsPath+DataSets.fileNames[rep]+".csv");
        
//      nearestNeighbourClassifiersTrainTest(DataSets.fileNames[rep],of,true);
//        nearestNeighbourClassifiersResample(DataSets.fileNames[rep],of,false,100);
    }
    

/** OLD CODE */

    
    public static void filteredNearestNeighbour(String resultsPath){
        DecimalFormat df = new DecimalFormat("###.###");
        OutFile of = new OutFile(resultsPath);
        System.out.println("************** EUCLIDEAN DISTANCE: All normalised/standardised*******************");
        System.out.println("\t\t 1NN \t Cross Val kNN,");
        of.writeLine("NNFilter, kNNFilter");
        for(int i=0;i<files.length;i++){
            try{
                    Instances test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TEST");
                    Instances train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TRAIN");			
                    if(!(files[i].equals("ElectricDevices")|| files[i].equals("Herring"))){
                        NormalizeCase norm=new NormalizeCase();
                        norm.setNormType(NormalizeCase.NormType.STD_NORMAL);                              
                      test=norm.process(test);
                      train=norm.process(train);
                    }
//Filter to 50% of the data set with info gain                                
                    AttributeSelection as = new AttributeSelection(); 
                    Ranker r= new Ranker();
                    r.setNumToSelect((train.numAttributes()-1)/2);
                    as.setSearch(r);
                    as.setEvaluator(new InfoGainAttributeEval());
                    as.SelectAttributes(train);
                    Instances trainSmall=as.reduceDimensionality(train);
                    Instances testSmall=as.reduceDimensionality(test);
                    Classifier a=new IBk(1);
                    kNN b= new kNN(100);
                    b.setCrossValidate(true);
                    b.normalise(false); 
                    a.buildClassifier(trainSmall);
                    b.buildClassifier(trainSmall); 
                    double acc=utilities.ClassifierTools.accuracy(testSmall,a);
                   double acc2=utilities.ClassifierTools.accuracy(testSmall,b);
                    System.out.println(files[i]+"\t"+df.format((acc))+"\t"+df.format((acc2)));
                    of.writeLine(files[i]+","+df.format((acc))+","+df.format((acc2)));
            }catch(Exception e){
//			System.out.println(trainSmall.toSting());
                System.out.println(" Error with file+"+files[i]+" ="+e);
                e.printStackTrace();
                System.exit(0);
            }        
        }        
    }
    
    public static void ensembleNearestNeighbour(String resultsPath){
           DecimalFormat df = new DecimalFormat("###.###");
            OutFile of = new OutFile(resultsPath);
        System.out.println("************** EUCLIDEAN DISTANCE: All normalised/standardised*******************");
        System.out.println("\t\t 1NN \t Cross Val kNN,");
        of.writeLine("NNFilter, kNNFilter");
        for(int i=0;i<files.length;i++)
        {
            try{
                Instances test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TEST");
                Instances train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TRAIN");			
                //Bagging with 20 base classifiers
                int bagPercent=66;
                Bagging a=new Bagging();
                a.setClassifier(new kNN(1));
                a.setNumIterations(20);
                a.setBagSizePercent(bagPercent);
                //Bagging with 100 base classifiers
                Bagging b=new Bagging();
                b.setClassifier(new kNN(1));
                b.setNumIterations(50);
                b.setBagSizePercent(66);
                //Boosting with 20 base
                AdaBoostM1 c=new AdaBoostM1();
                c.setClassifier(new kNN(1));
                c.setNumIterations(20);
                c.setUseResampling(true);
                 AdaBoostM1 d=new AdaBoostM1();
                d.setClassifier(new kNN(1));
                d.setNumIterations(100);
                d.setUseResampling(true);
                a.buildClassifier(train);
                b.buildClassifier(train);
                c.buildClassifier(train);
                d.buildClassifier(train);
                double acc=utilities.ClassifierTools.accuracy(test,a);
                double acc2=utilities.ClassifierTools.accuracy(test,b);
                double acc3=utilities.ClassifierTools.accuracy(test,c);
                double acc4=utilities.ClassifierTools.accuracy(test,d);
                System.out.println(files[i]+"\t"+df.format((acc))+"\t"+df.format((acc2))+"\t"+df.format((acc3))+"\t"+df.format((acc4)));
                of.writeLine(files[i]+","+df.format((acc))+","+df.format((acc2))+","+df.format((acc3))+","+df.format((acc4)));
            }catch(Exception e){
//			System.out.println(trainSmall.toSting());
                System.out.println(" Error with file+"+files[i]+" ="+e);
                e.printStackTrace();
                System.exit(0);
            }        
        }        
    }     
    public static void simulatedTest(String resultsPath){ 
//Runs a simulated polynomial experiment  
        int runs=30;
        ShapeletModel[] s=new ShapeletModel[2];
        int nosCases=100;
        int[] casesPerClass={nosCases/2,nosCases/2};
        int nosClassifiers=6;
        //PARAMETER LIST:  numShapelets, seriesLength, shapeletLength, maxStart        

        OutFile of= new OutFile(resultsPath);
        of.writeLine("seriesLength,NN,kNN,NNFilter, kNNFilter,Bagging100,Boosting100");
        double[] sum=new double[nosClassifiers];
        double[] sumSq=new double[nosClassifiers];
        int start=50, end=500, inc=50;
        for(int seriesLength=start;seriesLength<=end;seriesLength+=inc){
            PolynomialModel[] p=new PolynomialModel[2];
            double[] powers={1,2,3};    //Cubic model
            double[] coeff1={1.0,-2.0,0.1};  
            double[] coeff2={1.0,-2.01,0.11};  
            p[0]=new PolynomialModel(powers,coeff1);
            p[1]=new PolynomialModel(powers,coeff2);
        
            for(int r=0;r<runs;r++){
    //Generate instances
                try{
                    DataSimulator ds=new DataSimulator(p);
                    Instances train=ds.generateDataSet(seriesLength,casesPerClass);
                    Instances test=ds.generateDataSet(seriesLength,casesPerClass);
        //Create classifiers
                    Classifier[] c =new Classifier[nosClassifiers];
                    c[0]=new kNN(1);
                    kNN b= new kNN(100);
                    b.setCrossValidate(true);
                    b.normalise(false); 
                    c[1]=b;
        //Filter to 50% of the data set with info gain                                
                    AttributeSelection as = new AttributeSelection(); 
                    Ranker ranker= new Ranker();
                    ranker.setNumToSelect((train.numAttributes()-1)/2);
                    as.setSearch(ranker);
                    as.setEvaluator(new InfoGainAttributeEval());
                    as.SelectAttributes(train);
                    Instances trainSmall=as.reduceDimensionality(train);
                    Instances testSmall=as.reduceDimensionality(test);
                    Classifier a=new IBk(1);
                    b= new kNN(100);
                    b.setCrossValidate(true);
                    b.normalise(false);   
                    c[2]=a;
                    c[3]=b;
    //Two ensembles
                    //Bagging with 1-0 base classifiers
                    int bagPercent=66;
                    Bagging bag=new Bagging();
                    bag.setClassifier(new kNN(1));
                    bag.setNumIterations(100);
                    bag.setBagSizePercent(bagPercent);
                    //Boosting with 100 base
                    AdaBoostM1 ada=new AdaBoostM1();
                    ada.setClassifier(new kNN(1));
                    ada.setNumIterations(100);
                    ada.setUseResampling(true);
                    c[4]=bag;
                    c[5]=ada;           
    //Train all classifiers
                    for(int j=0;j<c.length;j++){
                        if(j==2 || j==3)    //Use small data sets
                            c[j].buildClassifier(trainSmall);
                        else
                            c[j].buildClassifier(train);
                    }
    //Measure Accuracy
                    double[] acc = new double[nosClassifiers];
                    for(int j=0;j<c.length;j++){
                        if(j==2 || j==3)    //Use small data sets
                        acc[j]=utilities.ClassifierTools.accuracy(testSmall,c[j]);
                    else
                        acc[j]=utilities.ClassifierTools.accuracy(test,c[j]);                   
                    }                         
    //Update stats
                        System.out.print(" \t\t RUN :"+(r+1)+"\t");
                    for(int j=0;j<c.length;j++){
                        sum[j]+=acc[j];
                        sumSq[j]+=acc[j]*acc[j];
                        System.out.print(acc[j]+"\t");
                    }
                        System.out.print(" \n");
                }catch(Exception e){
                    System.out.println(" Error with simulated run ="+r);
                    e.printStackTrace();
                    System.exit(0);
                }
            }
            of.writeString(seriesLength+",");
            System.out.println("Series length = "+seriesLength+" accuracy");
    //Update stats
            for(int j=0;j<nosClassifiers;j++){
                sum[j]/=runs;   
                sumSq[j]=sumSq[j]/runs-sum[j]*sum[j];
            }
            for(int j=0;j<nosClassifiers-1;j++){
                System.out.println(sum[j]+" ("+sumSq[j]+") ");
                of.writeString(sum[j]+",");
            }
            of.writeLine(sum[nosClassifiers-1]+"");
        }
            
    }

    
    public static void kNearestNeighbour(String resultsPath){
                    
            DecimalFormat df = new DecimalFormat("###.###");
            OutFile of = new OutFile(resultsPath);
                    System.out.println("************** EUCLIDEAN DISTANCE: All normalised/standardised*******************");
                    System.out.println("\t\t 1NN \t Cross Val kNN,");
                    of.writeLine("Bk(1), Normalised/Standardised IBk(1)");
                    for(int i=0;i<files.length;i++)
                    {
                        try{
				Instances test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TEST");
				Instances train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TRAIN");			
                                NormalizeCase norm=new NormalizeCase();
                               
                                if((files[i].equals("ElectricDevices")))//Just standardise, file has a zero variance entry!
                                    norm.setNormType(NormalizeCase.NormType.STD);
                              else
                                    norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
                                test=norm.process(test);
                                train=norm.process(train);
				Classifier a=new IBk(1);
				kNN b= new kNN(100);
                                b.setCrossValidate(true);
                                b.normalise(false); 
                                a.buildClassifier(train);
                                b.buildClassifier(train); 
                                double acc=utilities.ClassifierTools.accuracy(test,a);
                               double acc2=utilities.ClassifierTools.accuracy(test,b);
				System.out.println(files[i]+"\t"+df.format((acc))+"\t"+df.format((acc2)));
                                of.writeLine(files[i]+","+df.format((acc))+","+df.format((acc2)));
        		}catch(Exception e){
			System.out.println(" Error with file+"+files[i]+" ="+e);
			e.printStackTrace();
			System.exit(0);
                        }        
                    }
        
    }
 
    public static void oneNearestNeighbour(String resultsPath){
                    
        DecimalFormat df = new DecimalFormat("###.###");
        String[] files=DataSets.fileNames;
        OutFile of = new OutFile(resultsPath);
        System.out.println("************** EUCLIDEAN DISTANCE: All normalised/standardised*******************");
        System.out.println("\t\t  IBk(1) \t Normalised/Standardised IBk(1)");
        of.writeLine("IBk(1),Normalised/Standardised IBk(1)");
        for(int i=0;i<files.length;i++)
        {
            try{
                    Instances test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TEST");
                    Instances train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TRAIN");			
                    Classifier b=new IBk(1);
                    b.buildClassifier(train);
                    double acc=utilities.ClassifierTools.accuracy(test,b);
                    double acc2;
                    NormalizeCase norm=new NormalizeCase();

                    if((files[i].equals("ElectricDevices"))){//Just standardise, file has a zero variance entry!
                        norm.setNormType(NormalizeCase.NormType.STD);
                        Instances test3=norm.process(test);
                        Instances train3=norm.process(train);
                        b.buildClassifier(train3);
                        acc2=utilities.ClassifierTools.accuracy(test3,b);
                    }
                  else{   
                        norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
                        Instances test2=norm.process(test);
                        Instances train2=norm.process(train);
                        b.buildClassifier(train2);
                        acc2=utilities.ClassifierTools.accuracy(test2,b);
                    }
                    System.out.println(files[i]+"\t"+df.format((acc))+"\t"+df.format((acc2)));
                    of.writeLine(files[i]+","+df.format((acc))+","+df.format((acc2)));
            }catch(Exception e){
            System.out.println(" Error with file+"+files[i]+" ="+e);
            e.printStackTrace();
            System.exit(0);
            }        
        }
    }
 
//My wrapper kNN has a attribute filter, but I'm not sure it works! Check with
// a small problem    
    
    public static void filterTest(){        
            Instances train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[58]+"\\"+files[58]+"_TEST");
            Instances test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+files[58]+"\\"+files[58]+"_TEST");
            kNN classifier = new kNN();
            AttributeSelection as = new AttributeSelection(); 
            try{
                Ranker r= new Ranker();
                r.setNumToSelect((train.numAttributes()-1)/4);
                as.setSearch(r);
                as.setEvaluator(new InfoGainAttributeEval());
                as.SelectAttributes(train);
                System.out.println("total number ="+(train.numAttributes()-1)+" number selected ="+as.numberAttributesSelected());
                int[] ranks=as.selectedAttributes();
                Instances trainSmall=as.reduceDimensionality(train);
                Instances testSmall=as.reduceDimensionality(test);
               System.out.println("Atts in new train ="+(trainSmall.numAttributes()-1)+" in testnumber selected ="+(testSmall.numAttributes()-1));
              System.out.println("TRAIN"+trainSmall);
                  
            }catch(Exception e){
			System.out.println(" Error ="+e);
			e.printStackTrace();
			System.exit(0);
            }
            
            kNN c3=new kNN(1);
		c3.setFilterAttributes(true);
		c3.setProportion(0.5);
           
    }



}
