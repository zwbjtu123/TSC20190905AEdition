/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package development;

import fileIO.OutFile;
import java.net.URL;
import java.net.URLClassLoader;
import java.text.DecimalFormat;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.functions.LibSVM;
import static weka.classifiers.functions.LibSVM.*;
import weka.classifiers.functions.OptimisedSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.*;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.*;
import static weka.core.TechnicalInformation.Field.URL;

/**
 *
 * @author ajb
 */
public class StandardClassification {
    
//Resampled test to see if finding k through cross validation improves performance
    public static void kNN_Test(){
        kNN knn = new kNN(1);
        IBk ib1= new IBk(1);
        knn.normalise(true);
        kNN knnCV= new kNN();
        knnCV.setCrossValidate(true);
        knnCV.setKNN(100);
        IBk ibk= new IBk(100);
        ibk.setCrossValidate(true);
        OutFile of = new OutFile(DataSets.uciPath+"uciKNNTest.csv");
        of.writeLine("problem,1NN,kNN");
        DecimalFormat df = new DecimalFormat("###.###");
        for(String s:DataSets.uciFileNames){
            Instances train=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-train");
            Instances test=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-test");
            try{
                knn.buildClassifier(train);
               ib1.buildClassifier(train);
                knnCV.buildClassifier(train);
              ibk.buildClassifier(train);
                double a1,a2=0,a3,a4=0;
                a1=ClassifierTools.accuracy(test, knn);
                a2=ClassifierTools.accuracy(test, ib1);
                a3=ClassifierTools.accuracy(test, knnCV);
                a4=ClassifierTools.accuracy(test, ibk);
                of.writeLine(s+","+a1+","+a2+","+a3+","+a4);
                System.out.println(s+","+df.format(a1)+","+df.format(a2)+","+df.format(a3)+","+df.format(a4));
                
            }catch(Exception e){
                System.out.println(" Exception builing a classifier");
                System.exit(0);
            }
            
        }
        
    }
 
    public static void kNN_Resampled(){
        kNN knn = new kNN(1);
        knn.normalise(true);
        kNN knnCV= new kNN();
        knnCV.setCrossValidate(true);
        knnCV.setKNN(100);
        int runs=100;
        OutFile of = new OutFile(DataSets.uciPath+"uciKNNResample.csv");
        of.writeLine("problem,1NN,kNN");
        DecimalFormat df = new DecimalFormat("###.###");
        for(String s:DataSets.uciFileNames){
            Instances train=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-train");
            Instances test=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-test");
            int trainSize=train.numInstances();
            int testSize=test.numInstances();
            Instances all=new Instances(train);
            all.addAll(test);
            double a1=0,a2=0;
            for(int i=0;i<runs;i++){
                try{
                    all.randomize(new Random());
                    train = new Instances(all);
                    test=new Instances(all,0);
                    for(int j=0;j<testSize;j++){
                        Instance ins=train.remove(0);
                        test.add(ins);
                    }
                    knn.buildClassifier(train);
                    knnCV.buildClassifier(train);
                    a1+=ClassifierTools.accuracy(test, knn);
                    a2+=ClassifierTools.accuracy(test, knnCV);
                }catch(Exception e){
                    System.out.println(" Exception builing a classifier");
                    System.exit(0);
                }
            }
            of.writeLine(s+","+(a1/runs)+","+(a2/runs));
            System.out.println(s+","+(df.format(a1/runs))+","+df.format(a2/runs));
        }
        
    }
/**Hypothesis: forming a heterogenous ensemble improves base classifiers on 
//   average
Test on 30 UCI sets. 
* */
    public static void ensembleTest(){
        OutFile of = new OutFile(DataSets.uciPath+"ensembleTest.csv");
        of.writeString("problem,");
        DecimalFormat df = new DecimalFormat("###.###");
        WeightedEnsemble we=new WeightedEnsemble();
        for(String str:we.getNames())
            of.writeString(str+",");
        of.writeString("\n");
        for(String s:DataSets.uciFileNames){
            Instances train=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-train");
            Instances test=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-test");
            try{
                we=new WeightedEnsemble();
                we.setWeightType("Proportional");
                we.buildClassifier(train);
                System.out.println(" Build finished");
                double[] weights=we.getWeights();
                double[][] preds=new double[test.numInstances()][];
                double[] ensemblePreds=new double[test.numInstances()];
                for(int i=0;i<test.numInstances();i++){
                    ensemblePreds[i]=we.classifyInstance(test.instance(i));
                    preds[i]=we.getPredictions();
                }
//  Measure accuracies
                double acc=0;
                double[] components=new double[preds[0].length];
                for(int i=0;i<test.numInstances();i++){
                    double tr=test.instance(i).classValue();
                    if(ensemblePreds[i]==tr)
                        acc++;
                    for(int j=0;j<preds[i].length;j++)
                        if(preds[i][j]==tr)
                            components[j]++;
                }
                acc/=test.numInstances();
                of.writeString(s+","+acc);
                System.out.print("\n"+s+","+df.format(acc));
                for(int j=0;j<components.length;j++){
                    components[j]/=test.numInstances();
                    of.writeString(","+components[j]);
                    System.out.print(","+df.format(components[j]));
                }
                of.writeString("\n");
             }catch(Exception e){
                System.out.println(" Exception builing a classifier");
                System.exit(0);
            }
        }
    }
    
    
    public static void svmTest(){
        OutFile of = new OutFile(DataSets.uciPath+"svmTest.csv");
        of.writeLine("problem,SMOL,SMOQ,SMORBF,LibSVM");
        DecimalFormat df = new DecimalFormat("###.###");
        for(String s:DataSets.uciFileNames){
            Instances train=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-train");
            Instances test=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-test");
            try{
                SMO svml=new SMO();
                PolyKernel kernel = new PolyKernel();
                kernel.setExponent(1);
                svml.setKernel(kernel);
                SMO svmq=new SMO();
                kernel = new PolyKernel();
                kernel.setExponent(2);
                svmq.setKernel(kernel);
                SMO svmrbf=new SMO();
                RBFKernel kernel2 = new RBFKernel();
                kernel2.setGamma(1/(double)(train.numAttributes()-1));
                svmrbf.setKernel(kernel2);
                LibSVM libsvm = new LibSVM();
                LibSVM libsvmL = new LibSVM();
                libsvmL.setKernelType(new SelectedTag(KERNELTYPE_LINEAR, TAGS_KERNELTYPE));
                LibSVM libsvmQ= new LibSVM();
                libsvmQ.setKernelType(new SelectedTag(KERNELTYPE_POLYNOMIAL, TAGS_KERNELTYPE));
                OptimisedSVM betterSVM=new OptimisedSVM();
                
                svml.buildClassifier(train);
                svmq.buildClassifier(train);
                svmrbf.buildClassifier(train);
//                libsvm.buildClassifier(train);
//                libsvmL.buildClassifier(train);
//                libsvmQ.buildClassifier(train);
                betterSVM.buildClassifier(train);
                double accL=0,accQ=0,accRBF=0;
               accL=ClassifierTools.accuracy(test, svml);
                accQ=ClassifierTools.accuracy(test, svmq);
                accRBF=ClassifierTools.accuracy(test,svmrbf);

//                double accLibSVM = ClassifierTools.accuracy(test, libsvm);
 //               double accLibSVML = ClassifierTools.accuracy(test, libsvmL);
 //               double accLibSVMQ = ClassifierTools.accuracy(test, libsvmQ);
                
                double myAcc = ClassifierTools.accuracy(test, betterSVM);
                of.writeLine(s+","+(1-accL)+","+(1-accQ)+","+(1-accRBF)+","+","+(1-myAcc));
                //+(1-accLibSVM)+","+(1-accLibSVML)+","+(1-accLibSVMQ)+","
                System.out.println(s+","+df.format(accL)+","+df.format(accQ)+","+df.format(accRBF)+","+","+df.format(myAcc));
                //df.format(accLibSVM)+","+df.format(accLibSVML)+","+df.format(accLibSVMQ)+
                
             }catch(Exception e){
                System.out.println(" Exception builing a classifier = "+e);
                System.exit(0);
            }
        }
    }
    
    public static void main(String[] args){
        
        ClassLoader cl = ClassLoader.getSystemClassLoader();
 
        URL[] urls = ((URLClassLoader)cl).getURLs();
 
        for(URL url: urls){
        	System.out.println(url.getFile());
        }
   
        //kNN_Test();
        //kNN_Resampled();
       //ensembleTest();
        svmTest();
    }
    
    
}
