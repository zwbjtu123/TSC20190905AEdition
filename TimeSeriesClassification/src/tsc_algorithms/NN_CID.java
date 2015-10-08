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
package tsc_algorithms;

import development.DataSets;
import java.util.Enumeration;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.DTW;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author ajb
 */
public class NN_CID  extends kNN{    
    CIDDistance cid=new CIDDistance();
    
       public NN_CID(){
           super();
           cid=new CIDDistance();
       }
 
    
    public void useDTW(){
        cid=new CIDDTWDistance();
//        if(b)
//            cid.baseDistance=new DTW();
 //       else
//            cid.baseDistance=new EuclideanDistance();
    }
    @Override
    public void buildClassifier(Instances train){      
        this.setDistanceFunction(cid);
//        cid.setInstances(train);
        super.buildClassifier(train);
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
    
}
