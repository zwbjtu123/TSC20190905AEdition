/*
Class to generate benchmark test split accuracies for UCR/UEA benchmark 
* Time Series Classification problems
* 
* oneNearestNeighbour:  These results should tally with those on the UCR website
* kNearestNeighbour: compare 1NN to kNN with k set through cross validation
 */
package development;


import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import statistics.simulators.DataSimulator;
import statistics.simulators.PolynomialModel;
import statistics.simulators.ShapeletModel;
import weka.attributeSelection.*;
import weka.classifiers.Classifier;
import utilities.ClassifierTools;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class NN_Benchmarks {
 
    static String[] files=TimeSeriesClassification.fileNamesTotalSizeSorted;
    public static void filteredNearestNeighbour(String resultsPath){
           DecimalFormat df = new DecimalFormat("###.###");
            OutFile of = new OutFile(resultsPath);
                    System.out.println("************** EUCLIDEAN DISTANCE: All normalised/standardised*******************");
                    System.out.println("\t\t 1NN \t Cross Val kNN,");
                    of.writeLine("NNFilter, kNNFilter");
                    for(int i=0;i<files.length;i++)
                    {
                        try{
				Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TEST");
				Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TRAIN");			
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
                            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TEST");
                            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TRAIN");			
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
    
    
    public static void kNearestNeighbour(String resultsPath){
                    
            DecimalFormat df = new DecimalFormat("###.###");
            OutFile of = new OutFile(resultsPath);
                    System.out.println("************** EUCLIDEAN DISTANCE: All normalised/standardised*******************");
                    System.out.println("\t\t 1NN \t Cross Val kNN,");
                    of.writeLine("Bk(1), Normalised/Standardised IBk(1)");
                    for(int i=0;i<files.length;i++)
                    {
                        try{
				Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TEST");
				Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TRAIN");			
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
            String[] files=TimeSeriesClassification.fileNamesTotalSizeSorted;
            OutFile of = new OutFile(resultsPath);
                    System.out.println("************** EUCLIDEAN DISTANCE: All normalised/standardised*******************");
                    System.out.println("\t\t  IBk(1) \t Normalised/Standardised IBk(1)");
                    of.writeLine("IBk(1),Normalised/Standardised IBk(1)");
                    for(int i=0;i<files.length;i++)
                    {
                        try{
				Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TEST");
				Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[i]+"\\"+files[i]+"_TRAIN");			
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
            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[58]+"\\"+files[58]+"_TEST");
            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+files[58]+"\\"+files[58]+"_TEST");
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
    public static void main(String[] args){
        simulatedTest("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\SimTes1.csv");
//        filterTest();
//       ensembleNearestNeighbour("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NN_Ensembles.csv");
//        filteredNearestNeighbour("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\NNFilters.csv");
//        oneNearestNeighbour("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\OneNN.csv");
//            kNearestNeighbour("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\kNN.csv");
         }
    
}
