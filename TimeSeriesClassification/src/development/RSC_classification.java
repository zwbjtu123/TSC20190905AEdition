/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import tests.TwoSampleTests;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.*;
import weka.classifiers.meta.*;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
Comparison of built in ensembles with RSC. Require
* Bagging
* Aaboost
* RandomComittee
* RandomSubSpace
* Dagging
* 
* missing:  Multiboost
*
* Ensemble size: 25 or 100
 */
public class RSC_classification {
   public static String dataPath="C:\\Users\\ajb\\Dropbox\\UCR Classification Problems\\";
   public static String resultPath="";
   public static String[] fileNames={"abalone",
                                        "waveform",
                                        "satimage",
                                        "banana",
                                        "ringnorm",
                                        "twonorm",
//image??
                                        "german",
                                        "wdbc",
                                        "yeast",
                                        "ionosphere",
                                        "sonar",
                                        "heart",
                                        "cancer",
                                        "wins",
                                        "ecoli"
/*                                        
                                        "clouds",
                                        "concentric",
                                        "diabetes",
                                        "glass2",
                                        "haberman",
                                        "liver",
                                        "magic",
                                        "pendigitis",
                                        "phoneme",
                                        "segment",
                                        "thyroid",
                                        "vehicle",
                                        "vowel",
  */
   }; 
    public static IteratedSingleClassifierEnhancer[] setEnsembleClassifiers(ArrayList<String> names){
            ArrayList<IteratedSingleClassifierEnhancer> sc2=new ArrayList<>();
            Classifier c;
            Bagging b=new Bagging();
            names.add("Bagging");
            sc2.add(b);
            AdaBoostM1 ada=new AdaBoostM1();
            names.add("Adaboost");
            sc2.add(ada);
            RandomSubSpace rs= new RandomSubSpace();
            names.add("RandomSubSpace");
            sc2.add(rs);
            RandomCommittee rc=new RandomCommittee();
            names.add("RandomCommittee");
            sc2.add(rc);
            MultiBoostAB mb=new MultiBoostAB();
            names.add("Multiboost");
            sc2.add(mb);
            
            IteratedSingleClassifierEnhancer[] sc=new IteratedSingleClassifierEnhancer[sc2.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=sc2.get(i);

            return sc;
   
    }
   
    public static Classifier[] setSingleLazyClassifiers(ArrayList<String> names){
            ArrayList<Classifier> sc2=new ArrayList<>();
            Classifier c;
            
//Lazy classifiers            
            c=new IBk(1);
            ((IBk)c).setCrossValidate(false);
            sc2.add(c);
            
            names.add("OneNN");
            c=new RandomizedSphereCover(1);
            sc2.add(c);
            names.add("OneRSC");
            IBk c2=new IBk(50);
            c2.setCrossValidate(true);
            sc2.add(c2);
            names.add("kNN");
            RandomizedSphereCover r=new RandomizedSphereCover();
            r.crossValidate(true);
            sc2.add(r);
            names.add("OneRSC");
/*            c=new KStar();
            sc2.add(c);
            names.add("KStar");
            c=new LWL();
            sc2.add(c);
            names.add("LWL");
  */          
            Classifier[] sc=new Classifier[sc2.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=sc2.get(i);

            return sc;
    } 
    
      public static Random rand=new Random();

      public static int estimateAlpha(Instances train){
        int alpha;
        int maxAlpha=10;
        int bestAlpha=1;
        double bestAcc=0;
        int folds=5;
        for(alpha=1;alpha<maxAlpha;alpha++){
            Classifier c=new RandomizedSphereCover(alpha);
            try{
                Evaluation e=new Evaluation(train);
                e.crossValidateModel(c, train,folds, rand);                            
                double acc=e.correct()/(double)train.numInstances();
                System.out.println(" alpha ="+alpha+" acc ="+acc);
                if(acc>bestAcc){
                    bestAcc=acc;
                    bestAlpha=alpha;
                }
                    
            }
            catch(Exception e){
                System.out.println(" Error ="+e);
               e.printStackTrace();
               System.exit(0);
               
            }
            
        }
        return bestAlpha;
    }
    
    
    public static void assessRSC_EnsembleClassifiers(String fileName, int nosBase){

//Test 1: just do bagging and boosting with 1-RSC and 1-NN
      ArrayList<String> names=new ArrayList<>();
      IteratedSingleClassifierEnhancer[] c1=setEnsembleClassifiers(names);
      int runs=30;
      double[][] a;
      rand.setSeed(100);
      OutFile of =new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\AccEnsemblelassifiers"+nosBase+".csv");
      OutFile of2 =new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\SDEnsembleClassifiers"+nosBase+".csv");
      of.writeString("\n");
      for(String s:names){
          of.writeString("RSC"+s+",");
          of2.writeString("RSC"+s+",");
      }
/*      for(String s:names){
          of.writeString("kNN"+s+",");
          of2.writeString("kNN"+s+",");
      }
      for(String s:names){
          of.writeString("Tree"+s+",");
          of2.writeString("Tree"+s+",");
          
      }
*/      of2.writeString("\n");
        try{
                for(int i=0;i<fileNames.length;i++)
                {
                    of2.writeString(fileNames[i]+",");
                    of.writeString(fileNames[i]+",");
                    System.out.println(" Problem = "+fileNames[i]);
                    c1=setEnsembleClassifiers(names);
                    double[] sum=new double[c1.length];
                    double[] sumsq=new double[c1.length];
                    Instances train=ClassifierTools.loadData(dataPath+fileNames[i]+"\\"+fileNames[i]+"-train");
                    Instances test=ClassifierTools.loadData(dataPath+fileNames[i]+"\\"+fileNames[i]+"-test");
                    Instances all=new Instances(train);
                    int testSize=test.numInstances();
                    for(int j=0;j<test.numInstances();j++)
                        all.add(test.instance(j));
                    
                    for(int j=0;j<runs;j++){
                        //Form randomised test train split
                        all.randomize(rand);
                        train=new Instances(all);
                        test=new Instances(all,0);
                        for(int k=0;k<testSize;k++){
                            Instance temp=train.instance(0);
                            test.add(temp);
                            train.delete(0);                            
                        }
                        //Estimate alpha parameter
                        int alpha=estimateAlpha(train);
                        System.out.println(" Run ="+j+" best Alpha ="+alpha);
                        //Set the classifiers
                        c1=setEnsembleClassifiers(names);
                        for(IteratedSingleClassifierEnhancer s:c1){
                            s.setClassifier(new RandomizedSphereCover(alpha));
                            s.setNumIterations(nosBase);
                        }   
                        //Build classifiers and evaluate test accuracy
                        for(int k=0;k<c1.length;k++){
                            c1[k].buildClassifier(train);
                            double acc=ClassifierTools.accuracy(test,c1[k]);
                            sum[k]+=acc;
                            sumsq[k]+=acc*acc;
                        }
                    }
                        //Store mean and variance over runs.
                    for(int k=0;k<c1.length;k++){
                        sum[k]/=runs;
                        sumsq[k]=sumsq[k]/runs-sum[k]*sum[k];
                        of.writeString(sum[k]+",");
                        of2.writeString(sumsq[k]+",");
                    }
                        of.writeString("\n");
                        of2.writeString("\n");
                    
                }
            }
        catch(Exception e){
                    System.out.println(" Error in accuracy ="+e);
                    e.printStackTrace();
                    System.exit(0);
        }      
      
//Subspace      
  }
  public static void assessSingleClassifiers(String fileName){
      ArrayList<String> names=new ArrayList<String>();
      Classifier[] c= setSingleLazyClassifiers(names);
      double[][] a;
      int folds=10;
      OutFile of =new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\CVAcc"+folds+"AccSingleClassifiers.csv");
      OutFile of2 =new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\CVAcc"+folds+"SingleClassifiers.csv");
      for(int i=0;i<names.size();i++){
          of.writeString(","+names.get(i));
          of2.writeString(","+names.get(i));
      }
        try{
                for(int i=0;i<fileNames.length;i++)
                {
                    of2.writeString(fileNames[i]+",");
                    of.writeString(fileNames[i]+",");
                    Instances train=ClassifierTools.loadData(dataPath+fileNames[i]+"\\"+fileNames[i]+"-train");
                    Instances test=ClassifierTools.loadData(dataPath+fileNames[i]+"\\"+fileNames[i]+"-test");
                    Instances all=new Instances(train);
                    for(int j=0;j<test.numInstances();j++)
                        all.add(test.instance(i));
                    all.randomize(new Random());
                    System.out.println(" Problem = "+fileNames[i]);
                    for(int j=0;j<c.length;j++){
//                        acc[i][j]=ClassifierTools.singleTrainTestSplitAccuracy(c[j], train, test);
                        a=ClassifierTools.crossValidationWithStats(c[j], all, folds);
                        System.out.println("\t\t"+names.get(j)+" acc ="+a[0][0]+" sd ="+a[1][0]);
                        of.writeString(a[0][0]+",");
                        of2.writeString(a[1][0]+",");
                    }
                        of.writeString("\n");
                        of2.writeString("\n");
                }
            }
        catch(Exception e){
                    System.out.println(" Error in accuracy ="+e);
                    e.printStackTrace();
                    System.exit(0);
        }
  }   
  
    public static void testIBK(){
        
        for(int i=0;i<fileNames.length;i++){
                    Instances train=ClassifierTools.loadData(dataPath+fileNames[i]+"\\"+fileNames[i]+"-train");
                    Instances test=ClassifierTools.loadData(dataPath+fileNames[i]+"\\"+fileNames[i]+"-test");
                    Instances all=new Instances(train);
                    for(int j=0;j<test.numInstances();j++)
                        all.add(test.instance(i));
                    all.randomize(new Random());
                    System.out.println(" Problem = "+fileNames[i]);
                    IBk[] c =new IBk[2];
                    c[0]=new IBk(1);
                    c[0].setDebug(true);
                    c[1]=new IBk();
                    c[1].setDebug(true);
                    int folds=2;
                    c[1].setCrossValidate(true);
                    try{
                            for(int j=0;j<c.length;j++){
                                Evaluation e=new Evaluation(all);
                                e.crossValidateModel(c[j], all, folds, new Random());
                                System.out.println(" Acc = "+e.correct()/all.numInstances());
                                System.out.println(" IB1 k ="+c[0].getKNN()+" IBk k="+c[1].getKNN());
                        }
                    }catch(Exception ex){
                            ex.printStackTrace();
                            System.exit(0);
                    }
        }
    }
  public static void findPairwiseStats(String file){
      InFile in=new InFile(file);
      int lines=in.countLines();
    in=new InFile(file);
      int nosClassifiers=5;
      String names=in.readLine();
      double[][] data=new double[nosClassifiers][lines-1];
      for(int i=0;i<lines-1;i++){
          names=in.readString();
          for(int j=0;j<nosClassifiers;j++)
              data[j][i]=in.readDouble();
      }
      double[] a=data[0];
        double[] b=data[1];
      TwoSampleTests ts=new TwoSampleTests();
      ts.performTests(a, b);
//T test 
      double tSig=tests.TwoSampleTests.studentT_TestStat(a,b);
//Robust Rank Sum
	double ranks=tests.TwoSampleTests.rrs_PValue(a,b);
//Mann-Whitney
	double mw=tests.TwoSampleTests.mw_PValue(a,b);
        System.out.println(" T test ="+tSig+"  Mann-Whitney ="+mw+" RRS "+ranks);
      
  }
   public static void main(String[] args){
       assessRSC_EnsembleClassifiers("EnsembleTest25.csv",25);
       assessRSC_EnsembleClassifiers("EnsembleTest100.csv",100);
//       findPairwiseStats("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\NonSubspaceAcc.csv");
//       testIBK();
  //      assessSingleClassifiers("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\RSC_Single_Lazy.csv");        
 //       assessEnsembleClassifiers("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\RSC_Ensemble.csv");
        
        
        
   }
}
