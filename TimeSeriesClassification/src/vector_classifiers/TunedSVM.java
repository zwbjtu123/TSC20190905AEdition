/*
Tony's attempt to see the effect of parameter setting on SVM.

Two parameters: 
kernel para: for polynomial this is the weighting given to lower order terms
    k(x,x')=(<x'.x>+a)^d
regularisation parameter, used in the SMO 

m_C
 */
package vector_classifiers;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ParameterSplittable;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import utilities.ClassifierResults;
import weka.core.*;

/**
 *
 * @author ajb
 
 TunedSVM sets the margin c through a ten fold cross validation.
 
 If the kernel type is RBF, also set sigma through CV, same values as c
 
 NOTE: 
 1. CV could be done faster?
 2. Could use libSVM instead
 * 
 */
public class TunedSVM extends SMO implements SaveParameterInfo, TrainAccuracyEstimate,SaveEachParameter{
    boolean setSeed=false;
    int seed;
    int min=-16;
    int max=16;
    double[] paraSpace;
    private static int MAX_FOLDS=10;
    private double[] paras;
    String trainPath="";
    boolean debug=false;
    boolean randomSearch=false; //If set to true the parameter space is randomly sampled
    int nosParamSettings=624;   //Randomly sampled this many times
    Random rng;
    ArrayList<Double> accuracy;
    private boolean kernelOptimise=false;   //Choose between linear, quadratic and RBF kernel
    private boolean paraOptimise=true;
    private ClassifierResults res =new ClassifierResults();
    
    
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
    @Override
    public void setPathToSaveParameters(String r){
            resultsPath=r;
            setSaveEachParaAcc(true);
    }
    @Override
    public void setSaveEachParaAcc(boolean b){
        saveEachParaAcc=b;
    }
    
    
    public TunedSVM(){
        kernelOptimise=false;
        kernel=KernelType.RBF;
        paraOptimise=true;
        setKernel(new RBFKernel());
        rng=new Random();
       accuracy=new ArrayList<>();
         
    }

    public void setSeed(int s){
        this.setSeed=true;
        seed=s;
        rng=new Random();
        rng.setSeed(seed);
    }
    
 @Override
    public void writeCVTrainToFile(String train) {
        trainPath=train;
    }    
//Think this always does para search?
//    @Override
//    public boolean findsTrainAccuracyEstimate(){ return findTrainAcc;}
    
    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into res.acc here
        return res;
    }     
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",C,"+paras[0];
        if(paras.length>1)
            result+=",Gamma,"+paras[1];
       for(double d:accuracy)
            result+=","+d;
        
        return result;
    }
    public enum KernelType {LINEAR,QUADRATIC,RBF};
    KernelType kernel;
    public void debug(boolean b){
        this.debug=b;
    }

    public void setKernelType(KernelType type) {
        switch (type) {
            case LINEAR:                     
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                setKernel(p);
            break;
            case QUADRATIC:
                PolyKernel p2=new PolyKernel();
                p2.setExponent(2);
                setKernel(p2);
            break;
            case RBF:
                RBFKernel kernel2 = new RBFKernel();
                setKernel(kernel2);
            break;
        }
    }
    
    public void setParaSpace(double[] p){
        paraSpace=p;
    }
    public void setStandardParas(){
        paraSpace=new double[max-min+1];
        for(int i=min;i<=max;i++)
            paraSpace[i-min]=Math.pow(2,i);
    }
    public void optimiseKernel(boolean b){kernelOptimise=b;}
    
    public boolean getOptimiseKernel(){ return kernelOptimise;}
    public void optimiseParas(boolean b){paraOptimise=b;}
    
    static class ResultsHolder{
        double x,y;
        ClassifierResults res;
        ResultsHolder(double a, double b,ClassifierResults r){
            x=a;
            y=b;
            res=r;
        }
    }

    public void tuneRBF(Instances train) throws Exception {
        paras=new double[2];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();

        double minErr=1;
        this.setSeed(rng.nextInt());
        
        Instances trainCopy=new Instances(train);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(folds);
        cv.buildFolds(trainCopy);
        
        
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        int count=0;
        OutFile temp=null;
        for(double p1:paraSpace){
            for(double p2:paraSpace){
                count++;
                if(saveEachParaAcc){// check if para value already done
                    File f=new File(resultsPath+count+".csv");
                    if(f.exists() && f.length()>0)
                        continue;//If done, ignore skip this iteration
                    else
                        temp=new OutFile(resultsPath+count+".csv");
                   if(debug)
                       System.out.println("PARA COUNT ="+count);
                }
                SMO model = new SMO();
                RBFKernel kern = new RBFKernel();
                kern.setGamma(p2);
                model.setKernel(kern);
                model.setC(p1);
                model.setBuildLogisticModels(true);
                
                tempResults=cv.crossValidateWithStats(model,trainCopy);

//                Evaluation eval=new Evaluation(temp);
//                eval.crossValidateModel(model, temp, folds, rng);
                double e=1-tempResults.acc;
                accuracy.add(tempResults.acc);

                if(debug)
                    System.out.println(" C= "+p1+" Gamma = "+p2+" Acc = "+(1-e));
                if(saveEachParaAcc){// Save to file and close
                    temp.writeLine(tempResults.writeResultsFileToString());
                    temp.closeFile();
                }                
                if(e<minErr){
                    minErr=e;
                    ties=new ArrayList<>();//Remove previous ties
                    ties.add(new ResultsHolder(p1,p2,tempResults));
                }
                else if(e==minErr){//Sort out ties
                    ties.add(new ResultsHolder(p1,p2,tempResults));
                }
            }
        }
        double bestC;
        double bestSigma;
        minErr=1;
        if(saveEachParaAcc){// Read them all from file, pick the best
            count=0;
            for(double p1:paraSpace){
                for(double p2:paraSpace){
                    count++;
                    tempResults = new ClassifierResults();
                    tempResults.loadFromFile(resultsPath+count+".csv");
                    double e=1-tempResults.acc;
                    if(e<minErr){
                        minErr=e;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new ResultsHolder(p1,p2,tempResults));
                    }
                    else if(e==minErr){//Sort out ties
                            ties.add(new ResultsHolder(p1,p2,tempResults));
                    }
//Delete the files here to clean up.
                    
                    File f= new File(resultsPath+count+".csv");
                    if(!f.delete())
                        System.out.println("DELETE FAILED "+resultsPath+count+".csv");
                }            
            }
        }
        ResultsHolder best=ties.get(rng.nextInt(ties.size()));
        bestC=best.x;
        bestSigma=best.y;
        paras[0]=bestC;
        setC(bestC);
        ((RBFKernel)m_kernel).setGamma(bestSigma);
        paras[1]=bestSigma;
        res=best.res;
        if(debug)
            System.out.println("Best C ="+bestC+" best Gamma = "+bestSigma+" best train acc = "+res.acc);
    }
    
   public void tunePolynomial(Instances train) throws Exception {
        paras=new double[1];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();
        double minErr=1;
        this.setSeed(rng.nextInt());
        
        Instances trainCopy=new Instances(train);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(folds);
        cv.buildFolds(trainCopy);
        
        
        ArrayList<ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;            
        int count=0;
        OutFile temp=null;
        
        for(double d: paraSpace){
            count++;
            if(saveEachParaAcc){// check if para value already done
                File f=new File(resultsPath+count+".csv");
                if(f.exists() && f.length()>0)
                    continue;//If done, ignore skip this iteration
                else
                    temp=new OutFile(resultsPath+count+".csv");
               if(debug)
                   System.out.println("PARA COUNT ="+count);
            }
            
            SMO model = new SMO();
            model.setKernel(m_kernel);
            model.setC(d);
            model.setBuildLogisticModels(true);
            
            tempResults=cv.crossValidateWithStats(model,trainCopy);
//                Evaluation eval=new Evaluation(temp);
//                eval.crossValidateModel(model, temp, folds, rng);
            double e=1-tempResults.acc;
            accuracy.add(tempResults.acc);
            if(saveEachParaAcc){// Save to file and close
                temp.writeLine(tempResults.writeResultsFileToString());
                temp.closeFile();
            }                
            if(e<minErr){
                minErr=e;
               ties=new ArrayList<>();//Remove previous ties
                ties.add(new ResultsHolder(d,0.0,tempResults));
            }
            else if(e==minErr){//Sort out ties
                ties.add(new ResultsHolder(d,0.0,tempResults));
            }
        }
        if(saveEachParaAcc){// Read them all from file, pick the best
            count=0;
            for(double p1:paraSpace){
                count++;
                tempResults = new ClassifierResults();
                tempResults.loadFromFile(resultsPath+count+".csv");
                double e=1-tempResults.acc;
                if(e<minErr){
                    minErr=e;
                    ties=new ArrayList<>();//Remove previous ties
                    ties.add(new ResultsHolder(p1,0.0,tempResults));
                }
                else if(e==minErr){//Sort out ties
                        ties.add(new ResultsHolder(p1,0.0,tempResults));
                }
//Delete the files here to clean up.

                File f= new File(resultsPath+count+".csv");
                if (f.exists()) {
                    System.out.println("SDAFHILGHLKAG");
                }
                if(!f.delete())
                    System.out.println("DELETE FAILED "+resultsPath+count+".csv");
            }  
        }        
        
        ResultsHolder best=ties.get(rng.nextInt(ties.size()));
        setC(best.x);
        res=best.res;
    }
     
    public void selectKernel(Instances train) throws Exception {
        KernelType[] ker=KernelType.values();
        double[] rbfParas=new double[2];
        double rbfCVAcc=0;
        double linearBestC=0;
        double linearCVAcc=0;
        double quadraticBestC=0;
        double quadraticCVAcc=0;
        for(KernelType k:ker){
            TunedSVM temp=new TunedSVM();
            Kernel kernel;
            switch(k){
                case LINEAR:                     
                    PolyKernel p=new PolyKernel();
                    p.setExponent(1);
                    temp.setKernel(p);
                    temp.setStandardParas();
                    temp.tunePolynomial(train);
                    linearCVAcc=temp.res.acc;
                    linearBestC=temp.getC();
                break;
                case QUADRATIC:
                    PolyKernel p2=new PolyKernel();
                    p2.setExponent(2);
                    temp.setKernel(p2);
                    temp.setStandardParas();
                    temp.tunePolynomial(train);
                    quadraticCVAcc=temp.res.acc;
                    quadraticBestC=temp.getC();
                break;
                case RBF:
                    RBFKernel kernel2 = new RBFKernel();
                    temp.setKernel(kernel2);
                    temp.setStandardParas();
                    temp.tuneRBF(train);
                    rbfCVAcc=temp.res.acc;
                    rbfParas[0]=temp.getC();
                    rbfParas[1]=((RBFKernel)temp.m_kernel).getGamma();
                    break;
            }
        }
//Choose best, inelligantly
        if(linearCVAcc> rbfCVAcc && linearCVAcc> quadraticCVAcc){//Linear best
            PolyKernel p=new PolyKernel();
            p.setExponent(1);
            setKernel(p);
            setC(linearBestC);
            paras=new double[1];
            paras[0]=linearBestC;
            res.acc=linearCVAcc;
        }else if(quadraticCVAcc> linearCVAcc && quadraticCVAcc> rbfCVAcc){ //Quad best
            PolyKernel p=new PolyKernel();
            p.setExponent(2);
            setKernel(p);
            setC(quadraticBestC);
            paras=new double[1];
            paras[0]=quadraticBestC;
            res.acc=quadraticCVAcc;
        }else{   //RBF
            RBFKernel kernel = new RBFKernel();
            kernel.setGamma(rbfParas[1]);
            setKernel(kernel);
            setC(rbfParas[0]);
            paras=rbfParas;
            res.acc=rbfCVAcc;
        }
    }
    
    
    @Override
    public void buildClassifier(Instances train) throws Exception {
        res.buildTime=System.currentTimeMillis();
        if(paraSpace==null)
            setStandardParas();
        if(kernelOptimise)
            selectKernel(train);
        else if(paraOptimise){
            if(this.getKernel() instanceof RBFKernel)
                tuneRBF(train); //Does MORE CV 
            else
                tunePolynomial(train);
        }
        
        super.buildClassifier(train);
        
        res.buildTime=System.currentTimeMillis()-res.buildTime;
        if(trainPath!=null && trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(train.relationName()+",TunedSVM,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeLine(res.writeInstancePredictions());
        }        
    }
    
    public static void jamesltest() {
        try{ 

            String dset = "zoo";
//            int fold = 0;
            Instances all=ClassifierTools.loadData("C:/UCI Problems/"+dset+"/"+dset);
            
            for (int fold = 0; fold < 30; fold++) {
                
            
                Instances[] split=InstanceTools.resampleInstances(all,fold,0.5);
                Instances train=split[0];
                Instances test=split[1];

                TunedSVM svml = new TunedSVM();
                svml.optimiseParas(true);
                svml.optimiseKernel(false);
                svml.setBuildLogisticModels(true);
                svml.setSeed(fold);                
                svml.setKernelType(TunedSVM.KernelType.LINEAR);
    //
    //            TunedSVM svmq = new TunedSVM();
    //            svmq.optimiseParas(true);
    //            svmq.optimiseKernel(false);
    //            svmq.setBuildLogisticModels(true);
    //            svmq.setSeed(fold);                
    //            svmq.setKernelType(TunedSVM.KernelType.QUADRATIC);
    //
    //            TunedSVM svmrbf = new TunedSVM();
    //            svmrbf.optimiseParas(true);
    //            svmrbf.optimiseKernel(false);
    //            svmrbf.setBuildLogisticModels(true);
    //            svmrbf.setSeed(fold);                
    //            svmrbf.setKernelType(TunedSVM.KernelType.RBF);

                System.out.println("\n\nTSVM_L:");
                svml.buildClassifier(train);
                System.out.println("C ="+svml.getC());
                System.out.println("Train: " + svml.res.acc + " " + svml.res.stddev);
                double accL=ClassifierTools.accuracy(test, svml);
                System.out.println("Test: " + accL);
    //
    //
    //            System.out.println("\n\nTSVM_Q:");
    //            svmq.buildClassifier(train);
    //            System.out.println("C ="+svmq.getC());
    //            System.out.println("Train: " + svmq.res.acc + " " + svmq.res.stddev);
    //            double accQ=ClassifierTools.accuracy(test, svmq);
    //            System.out.println("Test: " + accQ);
    //
    //            System.out.println("\n\nTSVM_RBF:");
    //            svmrbf.buildClassifier(train);
    //            System.out.println("C ="+svmrbf.getC());
    //            System.out.println("Train: " + svmrbf.res.acc + " " + svmrbf.res.stddev);
    //            double accRBF=ClassifierTools.accuracy(test, svmrbf);
    //            System.out.println("Test: " + accRBF);
            }
        }catch(Exception e){
            System.out.println("ffsjava");
            System.out.println(e);
            e.printStackTrace();
        }
    }
    
    
    public static void main(String[] args) {
//        jamesltest();
 //       System.exit(0);
        
        
        String sourcePath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
        String problemFile="ItalyPowerDemand";
        DecimalFormat df = new DecimalFormat("###.###");
        Instances all=ClassifierTools.loadData(sourcePath+problemFile+"/"+problemFile+"_TRAIN");
        Instances[] split=InstanceTools.resampleInstances(all,0,0.5);
        Instances train=split[0];
        Instances test=split[1];
        try{
            TunedSVM svml=new TunedSVM();
                svml.setPathToSaveParameters("C:\\Temp\\fold1_");
                svml.optimiseParas(true);
                svml.optimiseKernel(false);
                svml.setBuildLogisticModels(true);
                svml.setSeed(0);                
                svml.setKernelType(TunedSVM.KernelType.RBF);
                svml.debug=true;
/*            TunedSVM svmq=new TunedSVM();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            svmq.setKernel(kernel);
            TunedSVM svmrbf=new TunedSVM();
            RBFKernel kernel2 = new RBFKernel();
            kernel2.setGamma(1/(double)(all.numAttributes()-1));
            svmrbf.setKernel(kernel2);
            svmq.buildClassifier(train);
            System.out.println("BUILT QUAD");
            System.out.println(" Optimal C ="+svmq.getC());
           svmrbf.buildClassifier(train);
            System.out.println("BUILT RBF");
            System.out.println(" Optimal C ="+svmrbf.getC());
            double accL=0,accQ=0,accRBF=0;
           accQ=ClassifierTools.accuracy(test, svmq);
           accRBF=ClassifierTools.accuracy(test,svmrbf);

        
*/        
           svml.buildClassifier(train);
            System.out.println("BUILT LINEAR = "+svml);
            System.out.println(" Optimal C ="+svml.getC());
             
            double accL=ClassifierTools.accuracy(test, svml);

            System.out.println("ACC on "+problemFile+": Linear = "+df.format(accL)); //+", Quadratic = "+df.format(accQ)+", RBF = "+df.format(accRBF));
                
         }catch(Exception e){
            System.out.println(" Exception building a classifier = "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }

    
    protected class PolynomialKernel extends PolyKernel{
        double a=0; //Parameter
    @Override   
    protected double evaluate(int id1, int id2, Instance inst1)
        throws Exception {

        double result;
        if (id1 == id2) {
          result = dotProd(inst1, inst1);
        } else {
          result = dotProd(inst1, m_data.instance(id2));
        }
//Only change from base class        
        result += a;
        if (m_exponent != 1.0) {
          result = Math.pow(result, m_exponent);
        }
        return result;
      }

    
}
    
}
