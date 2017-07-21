/*
Implementation of the algorithm described in 

@inproceedings{batista11cid,
author="G. Batista and X. Wang and E. Keogh ",
title="A Complexity-Invariant Distance Measure for Time Series",
booktitle    ="Proceedings of the 11th {SIAM} International Conference on Data Mining (SDM)",
year="2011"
}
and 
@inproceedings{batista14cid,
author="G. Batista and E. Keogh and O. Tataw and X. Wang  ",
title="{CID}: an efficient complexity-invariant distance for time series",
  journal={Data Mining and Knowledge Discovery},
  volume={28},
  pages="634--669",
  year={2014}
}

The distance measure CID(Q,C)=ED(Q,C) × CF(Q,C), 
where ED is the Eucidean distance and
CF(Q,C) = max (CE(Q),CE(C))
          min (CE(Q),CE(C)) 
ie the ratio of complexities. In the paper, 

*/
package timeseriesweka.classifiers;

import development.DataSets;
import java.util.Enumeration;
import utilities.ClassifierTools;
import utilities.SaveParameterInfo;
import weka.classifiers.lazy.kNN;
import utilities.ClassifierResults;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import timeseriesweka.elastic_distance_measures.DTW;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author ajb
 */
public class NN_CID  extends kNN implements SaveParameterInfo{
     protected ClassifierResults res =new ClassifierResults();
    
    CIDDistance cid=new CIDDistance();
    
       public NN_CID(){
           super();
           cid=new CIDDistance();
       }
 
    
    public void useDTW(){
        cid=new CIDDTWDistance();
    }
    
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "G. Batista, E. Keogh, O. Tataw and X. Wang");
        result.setValue(TechnicalInformation.Field.YEAR, "2014");
        result.setValue(TechnicalInformation.Field.TITLE, "CID: an efficient complexity-invariant distance for time series");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "28");
        result.setValue(TechnicalInformation.Field.NUMBER, "3");
        result.setValue(TechnicalInformation.Field.PAGES, "634--669");
        return result;
      }

 //<editor-fold defaultstate="collapsed" desc="problems used in DAMI paper">   
    public static String[] problems={
        "FiftyWords",
        "Adiac",
        "Beef",
        "CBF",
        "ChlorineConcentration",
        "CinCECGtorso",
        "Coffee",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "ECG200",
        "ECGFiveDays",
        "FaceAll",
        "FaceFour",
        "FacesUCR",
        "Fish",
        "GunPoint",
        "Haptics",
        "InlineSkate",
        "ItalyPowerDemand",
        "Lightning2",
        "Lightning7",
        "Mallat",
        "MedicalImages",
        "Motes",
        "OSULeaf",
        "OliveOil",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "StarLightCurves",
        "SwedishLeaf",
        "Symbols",
        "SyntheticControl",
        "Trace",
        "TwoLeadECG",
        "TwoPatterns",
        "Wafer",
        "WordsSynonyms",
        "Yoga",
        "uWaveGestureLibraryX",
        "uWaveGestureLibraryY",
        "uWaveGestureLibraryZ"
    };
      //</editor-fold>  
    

//<editor-fold defaultstate="collapsed" desc="ACCURACY for CID DTW reported in DAMI paper">        
    static double[] reportedResults={
        0.7736,
        0.6215,
        0.5333,
        0.9989,
        0.6487,
        0.9457,
        0.8214,
        0.7513,
        0.8026,
        0.7949,
        0.9346,
        0.8900,
        0.7816,
        0.8556,
        0.8750,
        0.8985,
        0.8457,
        0.9267,
        0.4286,
        0.4145,
        0.9563,
        0.8689,
        0.7397,
        0.9254,
        0.7421,
        0.7955,
        0.6281,
        0.8333,
        0.8153,
        0.8772,
        0.9343,
        0.8832,
        0.9407,
        0.9733,
        0.9900,
        0.8622,
        0.9958,
        0.9945,
        0.7571,
        0.8443,
        0.7889,
        0.7217,
        0.7066    
    };
      //</editor-fold>  
    
    @Override
    public String getParameters() {
        return "BuildTime,"+res.buildTime;
    }
    
    
    @Override
    public void buildClassifier(Instances train){      
        res.buildTime=System.currentTimeMillis();
        this.setDistanceFunction(cid);
//        cid.setInstances(train);
        super.buildClassifier(train);
        res.buildTime=System.currentTimeMillis()-res.buildTime;
        
    }
    public static class CIDDistance extends EuclideanDistance {
   
        @Override
        public double distance(Instance one, Instance two){
            return this.distance(one, two, Double.MAX_VALUE);
        }

        @Override
        public double distance(Instance one, Instance two, double cutoff, PerformanceStats stats){
            return this.distance(one,two,cutoff);
        }

        @Override
        public double distance(Instance first, Instance second, double cutoff){
            
            double d=0;
//Find the acf terms
            double d1=0,d2=0;
            double[] data1=first.toDoubleArray();
            double[] data2=second.toDoubleArray();
            for(int i=0;i<first.numAttributes()-1;i++)
                d+=(data1[i]-data2[i])*(data1[i]-data2[i]);
            d=Math.sqrt(d);
            for(int i=0;i<first.numAttributes()-2;i++)
                d1+=(data1[i]-data1[i+1])*(data1[i]-data1[i+1]);
            for(int i=0;i<first.numAttributes()-2;i++)
                d2+=(data2[i]-data2[i+1])*(data2[i]-data2[i+1]);
            d1=Math.sqrt(d1+0.001); //This is from theircode
            d2=Math.sqrt(d2+0.001); //This is from theircode
            if(d1<d2){
                double temp=d1;
                d1=d2;
                d2=temp;
            }
            d=Math.sqrt(d);
            d=d*(d1/d2);
            return d;
        }
        
        
    }
    
    public static class CIDDTWDistance extends CIDDistance {
        DTW dtw = new DTW();
   
        @Override
        public double distance(Instance one, Instance two){
            return this.distance(one, two, Double.MAX_VALUE);
        }

        @Override
        public double distance(Instance one, Instance two, double cutoff, PerformanceStats stats){
            return this.distance(one,two,cutoff);
        }

        @Override
        public double distance(Instance first, Instance second, double cutoff){
            
            double d=0;
//Find the acf terms
            double d1=0,d2=0;
            double[] data1=first.toDoubleArray();
            double[] data2=second.toDoubleArray();
            
            d=dtw.distance(first, second);
            for(int i=0;i<first.numAttributes()-2;i++)
                d1+=(data1[i]-data1[i+1])*(data1[i]-data1[i+1]);
            for(int i=0;i<first.numAttributes()-2;i++)
                d2+=(data2[i]-data2[i+1])*(data2[i]-data2[i+1]);
            d1=Math.sqrt(d1)+0.001; //This is from theircode
            d2=Math.sqrt(d2)+0.001; //This is from theircode
            if(d1<d2){
                double temp=d1;
                d1=d2;
                d2=temp;
            }
            d=d*(d1/d2);
            return d;
        }
        
        
    }
    
    public static void recreateDTWDistance(){
        int c=0;
        for(String s:DataSets.ucrNames){
            kNN k= new kNN(1);
            NN_CID k2= new NN_CID();
            k2.useDTW();
            Instances train=ClassifierTools.loadData(DataSets.problemPath+s+"\\"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(DataSets.problemPath+s+"\\"+s+"_TEST");
            k.buildClassifier(train);
            k2.buildClassifier(train);
            double a1=ClassifierTools.accuracy(test, k);
            double a2=ClassifierTools.accuracy(test, k2);
            System.out.println(s+","+a1+","+a2);
            if(a2>a1)
                c++;
        }
        System.out.println("CID Better on "+c+" out of "+DataSets.ucrNames.length);
    }
    
    public static void recreateEuclideanDistance(){
        int c=0;
        for(String s:DataSets.ucrNames){
            kNN k= new kNN(1);
            NN_CID k2= new NN_CID();
            Instances train=ClassifierTools.loadData(DataSets.problemPath+s+"\\"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(DataSets.problemPath+s+"\\"+s+"_TEST");
            k.buildClassifier(train);
            k2.buildClassifier(train);
            double a1=ClassifierTools.accuracy(test, k);
            double a2=ClassifierTools.accuracy(test, k2);
            System.out.println(s+","+a1+","+a2);
            if(a2>a1)
                c++;
        }
        System.out.println("CID Better on "+c+" out of "+DataSets.ucrNames.length);
    }
    public static void main(String[]args){
        recreateEuclideanDistance();
//        recreateDTWDistance();
    }
    int[][] DTWOptimalWindows={
        {4,0,1,0,1,2,0,1,1,0,1,1,0,2,1,0,1,1,0,1,1,1,1,1,1,1,1,0,0,1,4,0,0,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,3,1,0,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,2,0,4,1,1,4,0,1,1,1,0,4,1,6,0,0,0,5,0,1,4,0,1,1,1,1,1,3,1,0,1,1},
        {0,0,0,0,0,0,0,0,0,0,0,1,3,2,14,0,14,0,1,1,0,0,1,0,1,0,0,1,0,0,3,0,0,1,2,0,1,1,0,1,1,0,1,0,1,2,0,0,2,2,0,0,2,0,0,0,1,0,0,3,2,0,0,0,0,4,0,0,0,0,0,0,0,1,3,2,0,1,1,0,0,1,0,0,1,3,0,0,0,0,8,0,0,0,0,0,1,1,0,1},
        {0,1,7,1,0,0,3,1,0,1,1,0,1,1,1,0,1,2,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,2,0,0,1,0,0,0,3,1,3,1,0,0,1,0,0,1,0,0,0,1,1,0,0,0,3,1,0,0,0,0,1,1,1,1,2,1,0,1,1,1,1,0,1,2,0,0,1,0,1,0,0,0,1,1,1,0,1,1},
        {8,6,6,9,6,9,9,2,23,6,9,20,4,8,4,4,8,8,2,8,6,9,11,9,6,2,9,5,8,8,9,4,8,6,8,2,5,14,0,8,10,3,7,8,2,2,11,6,9,5,7,6,5,0,6,9,9,3,9,9,9,7,7,8,13,2,10,8,9,8,9,9,6,9,5,9,9,6,9,2,7,9,3,2,9,7,9,11,6,9,3,7,4,8,0,8,0,8,6,10},
        {7,15,6,8,7,5,7,11,8,1,9,5,7,8,4,5,8,9,4,6,6,5,9,5,7,3,3,5,7,9,7,0,9,7,5,2,5,10,0,7,6,8,8,6,7,5,6,6,6,9,9,11,7,4,9,7,5,8,9,7,5,8,15,13,9,7,9,8,11,6,8,10,6,5,5,8,10,6,8,11,9,4,5,4,5,8,9,6,2,8,5,6,5,8,10,2,8,8,4,8},
        {1,0,0,4,0,0,2,3,1,0,2,0,4,1,6,2,1,2,1,0,0,4,1,1,2,4,0,1,0,4,4,1,0,2,2,4,1,2,3,2,2,13,0,1,1,3,7,0,3,2,0,1,0,0,1,5,1,5,1,1,0,1,0,1,0,1,0,1,2,4,1,6,0,1,0,10,2,3,0,1,0,0,6,1,0,0,1,0,0,2,1,0,0,4,0,4,4,1,1,1},
        {11,8,0,7,6,15,4,4,20,12,7,5,9,8,7,21,17,14,7,8,5,9,5,3,7,5,7,9,8,10,9,9,13,7,5,3,13,6,6,8,11,14,9,8,8,4,8,6,1,5,15,12,9,4,8,1,9,9,18,15,8,8,15,4,18,11,5,7,6,10,8,15,7,5,6,7,9,12,11,3,18,8,7,5,4,2,9,4,8,3,2,5,13,7,12,5,2,7,8,8},
        {0,0,0,0,0,0,0,0,1,0,0,0,1,0,2,0,0,0,0,0,2,0,2,0,0,0,0,11,0,1,0,0,18,0,2,0,0,0,0,0,0,4,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,2,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,2,0,0,0,1,2,0,1,0,0,0,0,0},
        {1,1,1,2,1,1,2,1,1,0,1,0,1,1,2,1,1,1,1,2,2,2,0,2,0,2,1,2,1,1,3,2,1,1,0,1,2,2,1,1,2,0,2,4,3,1,1,1,1,2,1,1,1,2,2,0,3,0,2,1,0,1,0,1,0,0,2,0,2,2,2,1,4,1,2,1,1,2,1,0,1,1,2,2,1,1,2,1,3,1,1,2,1,1,2,2,1,1,1,1},
        {0,0,3,0,0,0,0,0,0,0,0,0,0,0,2,0,0,3,0,0,0,3,0,0,4,3,0,0,0,0,3,0,0,0,0,0,0,3,2,4,0,0,0,0,3,0,1,0,0,0,0,0,0,2,0,0,0,0,3,3,4,0,0,2,4,0,0,0,0,0,3,0,2,0,0,3,3,0,0,0,0,0,0,3,0,2,0,0,0,0,0,0,1,4,4,0,4,0,0,0},
        {13,32,17,38,32,32,55,42,37,13,54,52,62,38,41,62,22,17,22,71,20,8,52,45,49,46,32,20,38,13,21,8,20,23,7,39,17,12,21,20,18,48,32,21,10,17,21,32,31,47,17,48,19,43,22,45,26,12,20,40,25,20,32,34,12,64,40,34,48,91,26,13,22,54,86,11,21,86,86,36,31,46,32,20,21,25,17,40,40,18,21,10,11,54,9,86,21,20,12,47},
        {8,10,6,10,11,9,14,8,9,13,13,21,14,14,12,11,10,6,8,11,7,8,10,14,15,5,9,7,5,7,8,13,9,5,10,13,9,5,13,11,11,11,7,12,7,11,14,9,9,7,7,8,7,6,8,13,13,13,16,8,14,11,7,11,10,8,4,11,10,6,8,10,8,10,11,10,10,8,6,6,8,11,7,13,14,7,10,8,7,11,7,11,10,11,6,14,9,8,11,7},
        {16,11,10,6,9,7,11,11,16,10,6,7,5,11,6,10,20,11,13,10,10,14,11,10,8,14,8,12,12,7,11,6,12,10,11,12,13,14,9,9,12,11,9,8,20,13,14,10,4,5,15,8,12,8,11,13,8,6,8,11,8,12,22,9,9,10,12,5,22,17,11,10,8,22,7,15,7,7,11,9,11,12,7,10,11,10,10,10,14,10,5,6,9,11,11,16,11,13,6,10},
        {5,7,5,13,11,7,17,9,8,6,8,16,12,11,5,5,5,12,7,12,10,8,12,5,6,10,7,8,4,8,7,11,6,16,16,7,7,10,13,15,5,6,14,9,7,8,15,5,14,15,7,8,14,7,12,12,14,8,6,8,13,12,10,11,12,12,10,14,11,6,10,12,5,8,5,5,7,7,12,8,15,7,4,8,9,9,7,8,9,9,8,9,5,9,5,10,10,16,6,5},
        {0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {2,4,0,2,3,3,2,2,2,4,3,6,3,2,6,2,8,3,3,6,3,2,3,3,3,3,6,9,8,4,3,2,2,2,4,0,3,3,3,2,3,6,2,2,8,2,9,3,6,2,3,8,6,0,3,3,9,3,0,3,3,3,2,3,2,2,2,3,6,8,2,3,3,2,3,2,2,2,2,8,3,0,3,3,6,2,3,2,4,6,2,0,2,7,2,2,6,4,4,2},
        {0,0,4,6,2,6,6,6,2,7,7,3,3,3,0,3,6,6,4,7,2,0,0,0,0,3,6,3,3,3,6,3,3,0,8,6,3,6,6,6,4,7,4,2,6,9,0,0,7,6,4,0,3,0,0,4,6,3,0,0,0,3,0,3,6,4,3,0,9,3,4,0,6,9,3,3,7,4,6,0,0,0,3,7,0,2,6,3,6,9,9,6,6,3,2,6,0,0,3,0},
        {0,0,0,0,0,0,0,3,0,0,0,3,0,0,0,0,0,4,4,0,4,0,4,4,0,0,0,6,0,0,0,0,0,0,3,6,3,3,0,0,0,0,4,4,0,0,0,0,0,3,0,0,3,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,4,0,0,0,0,3,0,0,0,3,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0},
        {7,12,4,4,17,20,4,3,24,4,1,4,7,17,4,4,4,6,6,5,4,0,22,18,4,24,6,4,8,4,8,13,3,16,4,4,15,4,4,6,4,14,4,4,20,4,4,17,7,10,20,4,19,7,7,4,4,1,0,6,4,24,4,4,4,3,3,4,7,20,4,28,7,4,4,15,4,7,15,4,17,4,4,7,0,20,17,4,14,4,15,4,4,6,6,20,4,4,15,3},
        {0,0,0,0,0,0,0,7,3,0,0,0,0,0,0,0,0,2,2,0,0,0,2,2,0,3,2,3,2,2,0,3,0,0,2,0,0,22,2,0,0,2,0,0,3,2,2,0,2,0,2,0,0,0,0,2,0,0,0,2,0,2,0,0,2,0,2,0,2,3,2,2,0,4,2,0,0,0,2,0,2,2,5,0,3,2,5,0,0,5,2,0,0,0,2,3,15,0,0,0},
        {1,12,1,0,2,1,5,16,0,3,1,1,17,2,12,10,1,3,3,2,1,2,0,5,1,26,0,13,61,11,2,0,4,24,2,13,28,1,0,2,4,3,1,9,0,2,0,2,18,8,3,1,10,6,13,33,9,1,0,0,6,0,0,1,2,17,0,11,0,1,1,2,6,6,14,0,14,0,6,3,1,1,1,0,1,3,0,4,4,0,0,27,28,53,3,0,0,0,0,1},
        {0,2,0,1,0,1,2,3,2,0,1,1,1,0,2,0,0,1,1,0,1,3,2,1,2,0,1,1,0,1,0,1,1,0,1,1,1,1,1,3,1,1,1,0,0,1,1,1,1,0,2,1,1,1,0,2,1,0,1,1,0,1,3,1,1,2,1,2,1,4,1,2,1,0,1,0,0,3,3,0,0,2,2,1,6,0,6,2,0,5,1,1,0,1,1,3,1,2,4,5},
        {14,11,12,10,10,14,11,13,11,10},
        {4,4,3,6,3,3,4,3,4,4,3,4,2,5,3,2,5,3,2,6,9,5,2,3,3,5,4,5,3,5,3,3,3,3,4,4,4,5,4,4,3,3,4,2,3,3,4,4,4,3,4,2,3,3,2,4,4,5,5,3,3,3,2,3,2,4,3,5,4,3,3,4,4,2,4,5,2,4,3,5,4,6,3,4,3,4,3,5,6,5,5,4,4,4,2,3,4,3,4,4},
        {2,3,5,5,0,3,5,7,11,3,0,6,4,3,2,0,4,3,1,3,3,4,3,2,5,1,3,3,1,4,1,2,0,2,2,2,3,2,3,6,1,2,1,1,2,2,1,0,1,4,2,0,4,1,2,3,4,6,7,5,3,5,2,6,6,3,3,4,2,7,6,3,1,5,2,2,2,3,1,5,5,1,4,3,3,2,4,6,0,3,3,2,3,5,2,3,4,12,3,3},
        {3,4,11,5,5,4,4,4,4,4,6,6,3,4,5,4,3,4,4,6,3,4,3,5,3,4,4,13,5,4,2,4,8,4,4,6,6,6,11,4,5,4,6,4,4,3,6,5,5,5,3,6,4,12,4,14,3,5,6,5,4,4,4,5,3,3,8,3,8,6,8,3,4,5,7,4,3,8,4,4,4,4,3,4,6,10,4,5,5,4,6,6,3,5,3,7,5,4,4,5},
        {9,6,6,5,8,6,5,8,6,6,7,6,6,6,6,5,9,5,9,8,7,5,8,6,5,6,6,6,6,6,6,7,6,6,6,7,5,6,6,6,7,7,6,5,6,5,5,7,6,6,6,6,5,6,5,9,6,5,7,8,9,5,5,6,10,8,6,6,5,3,6,6,6,5,6,7,6,6,5,6,6,6,8,6,6,6,6,5,6,5,6,6,6,8,9,6,5,6,6,7},
        {4,0,2,1,1,1,1,2,1,1,2,2,2,1,1,1,1,1,0,1,2,1,1,1,1,1,0,1,1,2,1,1,2,1,1,2,1,0,2,6,1,0,1,1,0,0,4,0,4,1,1,1,6,2,1,1,1,2,2,0,1,1,2,1,1,1,1,3,1,4,0,1,2,1,2,1,2,1,2,1,1,1,1,2,1,5,1,1,3,1,1,1,1,0,1,1,1,1,1,0},
        {1,1,1,1,1,1,1,1,1,1},
        {1,1,1,1,1,1,1,2,1,1},
        {0,4,5,3,5,0,3,2,4,3,3,2,1,3,6,6,3,4,6,2,5,6,5,5,3,3,3,3,4,4,1,3,6,4,3,5,0,5,5,2,6,5,5,3,3,5,5,1,7,2,5,4,0,5,3,7,2,1,5,2,3,4,3,5,1,2,5,4,3,6,5,3,5,3,3,0,5,5,4,4,4,3,5,3,4,1,7,1,4,3,5,2,2,3,4,3,4,5,0,3},
        {0,1,2,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,2,0,0,4,2,0,0,0,2,0,0,0,1,2,0,1,0,0,1,0,0,1,0,0,2,0,2,0,0,1,0,0,1,1,1,0,0,2,0,0,1,0,0,7,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,6,0,0,0,0,2,0,0,0,0,2,1,4,0,0,0,0,0},
        {0,1,1,1,1,1,6,6,2,6},
        {2,2,3,3,4,3,3,17,3,0,5,3,4,2,30,5,1,3,5,4,4,3,4,2,3,3,0,3,1,2,8,4,5,1,6,5,4,5,7,0,2,3,9,3,2,3,2,6,8,6,6,5,5,3,3,0,4,5,5,6,2,2,4,0,5,1,4,2,2,4,0,4,4,2,9,3,3,8,3,3,5,1,2,2,6,4,7,4,3,0,9,1,6,5,6,5,4,5,6,6},
        {5,1,0,6,4,3,3,1,6,3,1,3,3,4,2,4,2,3,4,3,2,0,4,1,2,2,3,5,3,3,0,7,6,2,2,5,2,5,0,1,3,1,3,5,2,5,2,5,2,5,2,4,1,3,3,3,6,5,2,1,5,0,0,5,2,5,4,2,3,2,2,0,1,1,8,3,4,3,4,1,2,2,5,2,4,3,5,3,1,1,8,3,8,5,2,5,3,3,2,3},
        {14,1,5,3,3,3,17,4,5,7,12,6,7,11,10,35,8,13,2,7,5,17,1,3,11,19,4,6,9,3,2,6,13,3,3,24,3,4,3,7,2,7,4,4,14,9,21,8,14,0,5,2,3,11,4,9,2,2,3,5,12,5,4,3,14,6,3,4,4,4,4,2,9,20,10,4,9,5,15,4,12,13,3,6,4,5,8,4,7,8,5,10,10,21,2,3,13,11,4,5},
        {1,3,1,0,5,1,3,2,1,1,2,0,2,4,0,0,2,8,1,0,4,5,2,0,2,0,1,1,5,4,1,0,2,3,0,1,1,3,2,2,1,2,3,1,3,4,0,0,1,0,5,1,2,2,1,3,3,0,2,4,1,3,1,4,0,1,2,0,3,3,6,1,1,2,1,0,4,3,2,2,2,2,0,2,5,4,0,0,0,0,3,1,0,1,1,3,2,5,2,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,5,0,0,0,0,0,0,0,0,0,0,0,5,5,0,5,0,0,0,5,0,5,0,0,0,0,0,0,0,5,5,0,0,0,0,5,0,0,5,0,0,0,0,9,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,9,0,0,0,0,0,5,17,0,0,0,0,0},
        {94,65,30,94,46,31,42,52,38,41,65,46,73,94,51,86,29,40,49,19,23,79,34,49,51,27,53,49,39,21,20,48,49,36,94,24,43,49,19,86,86,33,38,69,95,39,46,35,79,22,39,29,94,29,51,24,44,29,50,29,69,42,80,94,21,95,21,49,96,39,29,67,45,40,41,94,35,39,42,77,67,29,69,38,51,77,67,46,39,27,53,47,69,95,44,34,24,40,95,20},
        {6,15,8,4,17,2,24,17,3,9,6,12,11,6,9,12,40,20,18,4,12,16,13,11,8,11,5,24,4,12,6,10,3,20,6,21,5,6,9,12,16,4,14,9,5,7,6,2,6,12,6,20,17,18,13,6,10,13,6,20,13,4,4,5,11,4,24,6,4,51,5,6,5,14,6,12,6,11,12,12,12,2,46,21,15,33,2,12,3,11,5,22,13,5,14,13,6,6,5,6},
        {5,5,11,5,6,18,16,13,17,18,9,5,4,6,8,3,6,7,6,5,2,6,5,7,5,6,6,11,3,8,5,4,6,3,4,9,12,8,12,8,3,17,4,7,7,14,6,4,8,7,3,6,6,4,12,15,16,16,13,11,11,4,18,12,4,3,5,11,38,16,11,24,15,6,7,8,5,8,4,5,13,12,5,18,12,9,7,4,5,6,6,5,35,7,11,6,4,18,18,4},
        {0,2,1,0,0,2,2,3,1,0,1,0,1,1,0,5,3,7,0,5,2,5,0,0,6,3,4,3,4,0,0,4,6,1,0,1,0,3,3,3,5,5,0,1,1,2,4,2,3,0,2,0,5,2,3,0,1,2,0,0,0,0,3,0,2,7,3,2,1,4,2,1,0,0,1,2,0,0,4,2,1,1,0,0,0,2,3,3,2,0,4,0,1,1,0,0,6,3,0,3},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,2,0,0},
        {21,14,9,8,11,10,13,22,10,10,11,4,8,2,8,8,15,3,12,4,4,11,9,8,5,3,9,14,18,8,12,18,29,6,12,3,10,9,6,10,10,7,4,18,8,3,7,10,12,43,12,17,16,5,10,10,9,21,10,6,10,8,13,9,2,11,11,8,11,5,10,12,5,9,4,3,12,10,6,18,27,17,22,28,13,10,6,10,5,8,18,10,8,12,17,2,13,7,17,10},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,2,0,2,0,2,0,2,2,2,0,0,2,0,3,0,0,0,2,0,0,0,2,0,8,8,2,2,2,0,2,2,2,2,0,7,2,0,8,0,0,2,0,8,0,4,0,2,2,0,2,0,2,2,2,0,2,2,2,0,0,2,0,0,0,0,0,2,0,2,2,2,0,3,2,0,2,8,3,0,0,2,0,2,0,0,0,2,0,0,0,0,0,0,0,0,8},
        {4,7,4,2,7,7,4,2,4,0,3,2,2,7,4,7,4,0,7,0,2,0,0,3,7,4,7,4,2,2,4,4,3,0,7,0,4,7,0,0,7,2,2,0,0,4,2,7,3,7,7,7,2,3,4,4,7,7,4,2,2,2,2,0,2,0,4,7,2,7,0,0,3,4,7,0,2,4,7,0,4,0,7,0,7,7,0,0,2,2,7,4,0,2,0,2,0,3,2,3},
        {2,8,0,2,2,11,4,8,6,11,5,0,0,9,16,0,52,0,0,3,0,4,0,8,5,2,0,17,0,2,2,8,24,3,0,0,9,14,17,0,9,3,27,0,0,0,11,5,0,10,0,10,0,9,2,6,9,3,0,14,0,5,12,4,0,9,2,0,3,8,0,3,16,0,0,4,3,2,15,0,0,0,4,0,30,2,0,3,6,6,0,9,16,17,18,10,10,4,4,0},
        {0,0,1,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,2,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
        {6,7,6,7,6,8,3,11,5,7,3,6,11,11,5,5,7,4,7,4,8,5,9,5,5,3,2,4,6,5,6,4,5,7,4,19,6,6,6,4,10,6,4,5,6,6,7,4,11,2,10,5,5,2,9,8,25,11,7,6,5,10,7,6,10,7,7,5,5,6,5,6,4,9,11,5,4,26,4,5,6,3,4,15,12,5,5,4,7,4,5,6,6,13,10,6,6,5,17,8},
        {0,4,0,2,0,0,2,0,0,0,0,0,2,0,0,2,2,2,0,0,0,0,0,0,2,2,0,0,0,0,0,0,2,0,0,2,2,0,2,0,0,2,0,0,0,0,4,0,0,0,0,0,2,0,2,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {14,13,15,14,21,21,9,26,19,36,23,9,5,9,5,13,14,22,17,9,28,7,4,21,10,55,29,9,8,17,11,19,9,20,14,33,12,9,4,27,9,9,6,12,14,7,5,7,20,7,14,11,12,5,17,23,15,9,20,4,10,5,13,5,7,29,19,57,8,19,6,8,14,25,6,9,3,14,25,27,10,5,2,27,53,9,3,14,13,22,8,8,9,4,1,11,26,4,6,9},
        {6,6,5,5,5,6,7,5,3,5,5,5,6,7,5,5,6,2,6,5,5,2,6,5,4,5,5,6,6,5,6,3,5,2,7,6,4,5,5,6,4,5,3,4,3,6,6,3,2,7,3,3,6,5,2,7,6,6,7,7,5,6,5,4,5,5,5,2,3,7,3,7,6,6,7,5,1,6,5,3,6,4,3,5,6,5,6,6,5,5,4,6,3,3,4,5,3,6,3,3},
        {2,0,2,3,2,2,2,0,2,0,3,0,2,2,3,4,2,0,0,4,2,0,0,2,2,0,0,0,2,0,2,0,2,0,2,2,0,2,2,0,0,2,3,3,0,4,3,2,2,2,2,0,4,0,2,3,2,2,2,0,2,0,2,0,0,2,0,6,3,2,3,0,2,2,3,2,2,0,0,0,3,3,0,2,2,4,2,2,2,0,2,2,2,0,4,0,0,0,2,2},
        {0,0,0,2,0,6,6,6,0,2,2,0,0,2,3,0,0,2,0,0,0,0,3,0,2,0,0,0,0,0,0,0,6,4,0,0,0,0,0,6,0,0,2,2,0,0,0,6,6,0,0,0,0,0,2,0,0,4,6,2,6,0,0,0,6,0,0,0,0,0,6,0,4,0,0,3,0,3,4,0,0,0,0,0,0,2,0,0,2,0,2,0,0,0,4,0,2,4,4,0},
        {3,2,4,3,4,3,3,3,4,3,4,4,4,2,4,3,0,4,2,3,3,2,4,3,6,0,4,4,6,3,4,0,4,0,3,4,3,4,2,2,2,2,2,2,2,2,4,2,3,3,0,3,2,2,4,2,3,4,3,0,0,3,4,3,3,0,3,4,0,4,2,2,4,2,3,2,3,2,3,0,4,4,3,2,2,3,4,0,4,3,3,3,2,3,2,3,3,3,2,3},
        {8,43,5,4,7,5,8,23,10,36,37,7,4,2,5,4,5,5,36,22,23,5,8,6,5,6,7,4,7,7,10,5,2,9,8,22,10,5,9,5,7,5,22,7,9,28,7,5,23,9,20,43,5,23,7,7,3,5,5,43,4,9,4,5,5,6,5,5,8,4,4,33,8,5,5,3,5,4,5,6,40,4,5,5,9,5,15,24,4,7,23,5,7,21,5,7,4,7,8,22},
        {17,2,39,33,4,13,29,2,31,69,38,13,4,5,2,9,29,21,30,1,42,6,1,5,1,18,18,42,13,38,11,16,11,16,4,30,29,13,39,12,77,11,12,16,44,9,1,27,10,7,16,32,21,38,12,21,1,4,23,33,2,8,9,32,57,9,16,10,3,16,2,37,46,28,10,19,9,2,1,4,13,11,15,13,11,2,22,17,22,23,33,11,21,10,15,6,2,1,33,1},
        {2,1,3,4,12,7,0,13,3,4,4,0,6,1,6,3,1,1,3,0,8,3,2,1,1,0,3,4,1,2,4,8,8,4,4,5,3,2,8,3,5,2,13,4,0,1,3,3,4,9,2,2,7,1,3,3,3,10,13,6,7,3,8,3,4,10,1,5,2,3,8,5,6,11,3,3,4,5,9,2,0,4,7,3,3,11,7,6,1,1,12,3,6,15,4,3,3,1,3,12},
        {4,5,8,4,3,5,5,6,8,5,5,4,7,4,5,6,5,7,6,9,6,8,5,4,5,6,4,7,6,7,7,5,4,4,5,9,5,5,5,9,7,4,6,5,4,4,7,5,2,4,8,4,5,8,6,5,5,5,5,5,5,5,5,9,7,4,4,8,8,3,4,5,5,8,5,5,5,4,4,4,5,8,6,6,5,9,5,4,4,7,6,8,6,6,4,4,4,4,7,7},
        {15,6,32,6,13,5,22,15,24,13,30,20,13,16,13,13,16,4,12,20,13,12,21,5,43,10,17,13,22,34,12,3,38,6,23,8,20,17,29,13,13,23,17,34,17,13,4,23,19,5,12,12,13,14,20,19,16,16,12,13,9,32,16,13,12,25,16,6,20,12,6,13,20,5,23,16,21,5,8,12,6,16,13,17,7,13,19,22,13,20,20,5,16,16,23,6,13,13,8,13},
        {0,6,0,0,0,0,2,0,2,3,3,2,3,0,21,0,2,0,2,2,0,6,2,9,2,3,11,0,2,0,0,0,3,0,0,9,0,5,0,2,3,0,5,2,0,3,3,2,2,0,2,5,2,0,6,5,0,6,0,2,3,5,3,0,0,5,2,2,0,2,2,12,6,6,19,2,0,2,2,0,2,0,9,6,8,5,0,0,2,3,0,2,3,0,0,2,2,3,2,0},
        {0,0,4,11,0,4,2,0,2,4,0,4,0,0,0,0,14,7,0,0,0,4,2,0,0,2,2,2,0,0,16,22,10,0,0,0,11,2,0,2,5,24,17,0,16,0,0,2,2,8,22,19,2,5,4,4,2,2,2,2,2,0,0,21,8,0,8,2,4,0,0,0,19,17,4,2,13,7,0,4,4,2,14,0,0,14,17,2,2,0,5,4,4,7,11,7,2,0,2,0},
        {16,18,21,7,17,21,19,14,16,15},
        {0,0,11,0,0,0,12,1,1,0,0,1,0,4,0,11,0,0,10,3,5,1,1,0,0,0,0,11,9,12,1,1,0,12,1,1,2,0,0,1,0,0,0,0,0,0,0,0,12,3,1,0,2,12,1,1,8,0,0,0,0,0,9,1,1,0,0,12,1,1,0,9,1,0,0,12,0,0,0,0,0,0,0,1,12,8,1,0,0,0,0,1,1,0,8,1,6,0,12,1},
        {4,3,3,4,3,2,3,3,2,4,2,2,2,2,2,3,4,2,3,2,2,3,3,3,3,2,3,3,2,2,4,2,2,3,3,2,4,2,2,3,4,2,2,3,2,2,2,3,2,3,3,3,2,3,2,4,3,3,4,2,2,3,2,3,3,3,3,3,2,3,2,2,3,3,2,4,2,2,4,3,2,3,2,3,4,2,2,3,3,4,3,3,3,3,3,2,3,3,2,2},
        {8,9,1,14,0,7,8,4,0,0,13,3,4,2,7,0,6,0,10,4,3,16,6,11,6,4,2,5,6,5,9,4,2,5,2,6,1,5,9,4,7,3,7,9,4,17,0,6,19,5,4,0,18,8,4,4,4,1,18,4,2,9,9,14,4,3,0,7,5,0,15,0,13,11,3,17,0,5,12,3,2,7,9,0,10,7,5,12,11,1,7,6,11,4,7,9,10,0,9,7},
        {7,2,12,6,11,6,9,6,11,12,9,11,4,2,4,17,9,11,9,16,6,7,6,16,7,7,6,4,16,6,9,7,14,7,11,4,11,14,14,14,12,9,14,4,14,6,4,12,16,7,16,11,9,4,7,9,6,11,2,12,11,14,14,16,4,4,6,4,4,6,7,4,4,9,9,4,4,4,2,11,6,19,6,9,4,7,9,2,6,6,4,9,4,11,2,11,14,6,7,4},
        {8,12,5,11,4,17,12,6,11,20,18,11,10,25,10,13,4,17,17,13,18,4,9,15,6,25,24,13,7,13,5,19,17,11,18,50,11,12,6,11,6,17,11,17,4,13,14,6,19,30,15,15,8,23,11,0,10,17,23,0,4,12,11,27,5,15,5,16,12,7,12,14,6,16,8,7,12,15,7,8,11,20,15,5,14,5,12,10,28,0,16,18,12,29,7,9,23,18,11,12},
        {5,5,2,2,1,5,8,0,5,24,7,5,12,3,1,2,7,0,20,7,5,6,8,2,9,4,12,22,20,0,12,7,7,5,20,3,8,26,11,5,7,0,7,6,5,0,7,7,4,7,7,7,4,9,6,9,5,12,7,2,0,7,8,6,8,8,7,20,13,3,2,5,0,2,24,4,0,5,2,26,3,0,3,1,30,4,2,5,4,6,17,2,17,6,32,13,0,1,3,1},
        {4,3,3,4,6,3,3,4,5,5,4,7,5,5,5,6,4,3,3,8,5,5,4,3,3,4,3,3,5,3,4,4,5,3,3,11,5,3,3,4,11,8,3,4,4,4,6,5,5,5,5,4,4,3,5,5,11,7,6,6,6,3,5,4,11,6,4,5,3,4,4,4,3,6,3,6,6,3,19,6,5,3,4,3,5,4,4,4,5,4,6,19,4,5,4,4,31,6,4,3},
        {5,7,9,2,5,4,13,4,2,2,2,3,4,3,8,3,5,7,0,9,4,4,4,2,8,3,3,9,3,5,2,3,4,5,7,7,7,7,10,5,19,3,10,7,8,4,10,4,7,10,5,5,5,5,5,5,4,7,5,7,3,8,5,3,4,0,11,5,10,8,7,5,5,18,8,3,3,2,4,7,0,19,7,4,8,3,7,4,7,5,9,4,4,4,5,7,4,2,4,5},
        {5,8,7,5,4,5,5,7,10,7,8,7,6,6,6,5,5,8,8,5,5,8,6,9,7,6,11,6,7,9,4,8,5,6,9,8,8,10,8,8,7,6,7,7,8,7,8,6,5,9,7,8,11,4,9,8,6,6,6,8,5,8,6,6,6,4,6,6,9,8,6,6,9,7,6,5,6,6,8,9,5,9,8,10,5,9,7,8,6,6,6,5,9,6,8,6,5,7,6,10},
        {4,4,8,12,7,5,5,8,7,5,9,9,8,8,5,6,6,9,6,7,8,8,8,8,11,9,7,7,4,4,6,4,7,4,7,6,7,6,7,9,7,9,2,11,6,9,3,6,7,3,6,9,6,7,5,11,9,4,8,4,7,4,8,7,10,10,7,8,7,9,3,5,7,12,6,6,6,8,8,5,7,9,8,5,14,8,5,8,8,6,7,7,4,4,6,7,7,3,5,6},
        {5,6,5,6,8,5,5,4,8,6,9,5,4,4,8,4,8,4,6,8,3,7,4,3,6,8,3,3,4,7,3,4,5,7,5,7,4,3,4,6,4,6,5,6,10,11,4,4,6,6,5,5,3,5,6,6,5,5,6,5,4,1,4,7,7,8,13,7,9,5,3,8,7,5,7,8,6,7,5,6,6,7,5,4,4,1,3,9,6,4,7,5,10,5,3,5,3,6,4,7},
        {7,3,7,4,5,8,6,6,10,7,9,12,6,5,4,5,8,7,7,6,6,7,8,6,6,4,5,4,4,6,3,5,4,3,5,2,7,5,2,2,5,8,8,3,6,3,3,4,2,6,10,9,6,7,3,17,6,4,7,6,17,6,8,5,14,5,8,4,2,6,9,4,10,6,13,6,6,4,6,4,5,6,15,5,5,6,5,7,7,2,9,3,5,4,9,5,3,4,6,4},
        {4,6,3,3,3,7,3,4,4,5},
        {2,1,1,3,0,4,1,1,0,2,2,0,2,2,0,0,2,0,0,2,0,4,0,1,5,1,4,0,1,1,1,1,1,1,2,0,1,1,1,5,2,1,0,0,2,4,1,0,2,0,4,0,0,1,2,2,0,3,2,0,1,0,1,0,3,2,1,3,2,1,1,2,2,0,1,2,4,0,1,1,1,1,2,2,1,0,1,0,2,0,5,1,0,1,0,1,6,4,3,0},
        {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0},
        {9,7,4,11,6,6,6,10,3,5,5,6,6,6,7,3,4,4,5,5,6,4,6,6,8,9,10,6,5,8,7,7,6,5,5,9,6,5,7,6,6,11,8,8,6,8,6,8,6,8,9,6,12,11,4,6,5,10,7,10,6,10,10,6,15,13,6,9,7,6,6,6,9,6,6,3,3,5,8,9,14,11,10,4,8,7,6,9,7,6,10,6,10,11,6,10,6,6,7,6},
        {7,5,3,36,4,4,36,4,7,3,47,10,35,3,30,36,4,3,16,36,4,6,36,47,39,41,51,39,36,25,47,36,3,3,39,8,3,39,47,14,3,25,4,51,3,47,3,5,53,36,4,5,6,30,36,5,36,5,23,36,22,30,4,36,9,39,8,7,47,19,4,3,8,5,3,22,53,24,53,53,9,39,5,4,3,5,29,7,3,3,47,4,7,36,39,16,5,5,4,5},
        {7,9,3,9,3,5,24,30,4,24,47,6,3,6,53,23,36,3,16,22,4,7,51,3,47,4,4,39,4,53,47,47,53,23,34,30,9,16,4,4,3,53,4,16,4,23,4,4,25,3,4,8,14,22,24,36,24,4,8,4,23,34,21,21,13,39,4,3,39,23,7,37,12,39,3,39,41,4,4,53,3,36,24,3,35,7,23,4,4,3,3,30,35,46,47,6,5,24,4,4},
        {7,3,2,12,2,5,2,5,0,1,12,3,2,3,2,5,3,4,6,8,13,4,1,3,3,3,3,6,1,2,2,3,6,6,3,5,1,4,5,4,7,0,5,3,6,2,8,2,1,3,3,3,8,1,6,0,7,5,6,7,4,3,5,3,5,3,2,3,3,4,0,2,3,3,3,1,2,0,30,3,6,3,2,2,4,3,2,2,2,4,2,4,2,4,3,2,21,5,2,8},
        };
}
