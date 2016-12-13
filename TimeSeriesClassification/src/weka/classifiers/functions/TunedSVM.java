/*
Tony's attempt to see the effect of parameter setting on SVM.

Two parameters: 
kernel para: for polynomial this is the weighting given to lower order terms
    k(x,x')=(<x'.x>+a)^d
regularisation parameter, used in the SMO 

m_C
 */
package weka.classifiers.functions;

import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.*;

/**
 *
 * @author ajb
 
 TunedSVM sets the margin c through a ten fold cross validation.
 
 If the kernel type is RBF, also set sigma through CV, same values as c
 
 NOTE: 
 1. CV could be done faster with Gavin's trick
 2. Could use libSVM instead
 * 
 */
public class TunedSVM extends SMO implements SaveCVAccuracy {
    int min=-16;
    int max=8;
    double[] paraSpace;
    private static int MAX_FOLDS=10;
    private double[] paras;
    private double cvAcc=0;
    String trainPath;
    enum KernelType {LINEAR,QUADRATIC,RBF};
    KernelType kernel;
    public TunedSVM(){
        kernelOptimise=false;
        kernel=KernelType.RBF;
        paraOptimise=true;
        setKernel(new RBFKernel());
        
    }
    private boolean kernelOptimise=false;   //Choose between linear, quadratic and RBF kernel
    private boolean paraOptimise=true;
    public void optimiseKernel(boolean b){kernelOptimise=b;}
    public void optimiseParas(boolean b){paraOptimise=b;}
    public void buildRBF(Instances train) throws Exception {
        paras=new double[2];
        int folds=MAX_FOLDS;
        double minErr=1;
        double bestC=min;
        double bestSigma=min;
        for(int i=min;i<=max;i++){
            for(int j=min;j<=max;j++){
                Instances temp=new Instances(train);
                Evaluation eval;
                SMO model = new SMO();
                RBFKernel kernel = new RBFKernel();
                kernel.setGamma(Math.pow(2,j));
                model.setKernel(kernel);
                model.setC(Math.pow(2,i));
       
                eval = new Evaluation(train);
                eval.crossValidateModel(model, temp, folds,new Random());
                double e=eval.errorRate();
                if(e<minErr){
                    minErr=e;
                    bestC=i;
                    bestSigma=j;
                }
            }
        }
        paras[0]=Math.pow(2,bestC);
          setC(Math.pow(2,bestC));
          ((RBFKernel)m_kernel).setGamma(Math.pow(2,bestSigma));
        paras[1]=Math.pow(2,bestSigma);
        cvAcc=1-minErr;  
    }
    
 
   public void buildQuadratic(Instances train) throws Exception {
        paras=new double[1];
        int folds=MAX_FOLDS;
        double minErr=1;
        double bestC=min;
        for(int i=min;i<=max;i++){
            Instances temp=new Instances(train);
            Evaluation eval;
            SMO model = new SMO();
            model.setKernel(m_kernel);
            model.setC(Math.pow(2,i));
            eval = new Evaluation(train);
            eval.crossValidateModel(model, temp, folds,new Random());
            double e=eval.errorRate();
            if(e<minErr){
                minErr=e;
                bestC=i;
            }
        }
        setC(Math.pow(2,bestC));
        cvAcc=1-minErr;
        paras[0]=Math.pow(2,bestC);  
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
                    temp.buildQuadratic(train);
                    linearCVAcc=temp.cvAcc;
                    linearBestC=temp.getC();
                break;
                case QUADRATIC:
                    PolyKernel p2=new PolyKernel();
                    p2.setExponent(2);
                    temp.setKernel(p2);
                    temp.buildQuadratic(train);
                    quadraticCVAcc=temp.cvAcc;
                    quadraticBestC=temp.getC();
                break;
                case RBF:
                    RBFKernel kernel2 = new RBFKernel();
                    temp.setKernel(kernel2);
                    temp.buildRBF(train);
                    rbfCVAcc=temp.cvAcc;
                    rbfParas[0]=temp.getC();
                    rbfParas[1]=((RBFKernel)temp.m_kernel).getGamma();
                    break;
            }
        }
//Choose best, ineligantly
        if(linearCVAcc> rbfCVAcc && linearCVAcc> quadraticCVAcc){//Linear best
            PolyKernel p=new PolyKernel();
            p.setExponent(1);
            setKernel(p);
            setC(linearBestC);
            paras=new double[1];
            paras[0]=linearBestC;
            cvAcc=linearCVAcc;
        }else if(quadraticCVAcc> linearCVAcc && quadraticCVAcc> rbfCVAcc){ //Quad best
            PolyKernel p=new PolyKernel();
            p.setExponent(2);
            setKernel(p);
            setC(quadraticBestC);
            paras=new double[1];
            paras[0]=quadraticBestC;
            cvAcc=quadraticCVAcc;
        }else{   //RBF
            RBFKernel kernel = new RBFKernel();
            kernel.setGamma(rbfParas[1]);
            setKernel(kernel);
            setC(rbfParas[0]);
            paras=rbfParas;
            cvAcc=rbfCVAcc;
        }
    }
    
    
    @Override
    public void buildClassifier(Instances train) throws Exception {
        if(kernelOptimise)
            selectKernel(train);
        else if(paraOptimise){
            if(this.getKernel() instanceof RBFKernel)
                buildRBF(train); //Does MORE CV 
            else
                buildQuadratic(train);
        }
        
        super.buildClassifier(train);
    }
    public static void main(String[] args) {
        String sourcePath="C:\\Users\\ajb\\Dropbox\\UCI Problems\\";
        String problemFile="monks-1";
        DecimalFormat df = new DecimalFormat("###.###");
        Instances all=ClassifierTools.loadData(sourcePath+problemFile+"/"+problemFile);
        Instances[] split=InstanceTools.resampleInstances(all,0,0.5);
        Instances train=split[0];
        Instances test=split[1];
        try{
            TunedSVM svml=new TunedSVM();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            svml.setKernel(kernel);
            TunedSVM svmq=new TunedSVM();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            svmq.setKernel(kernel);
            TunedSVM svmrbf=new TunedSVM();
            RBFKernel kernel2 = new RBFKernel();
            kernel2.setGamma(1/(double)(all.numAttributes()-1));
            svmrbf.setKernel(kernel2);
           svml.buildClassifier(train);
            System.out.println("BUILT LINEAR = "+svml);
            System.out.println(" Optimal C ="+svml.getC());
             
            svmq.buildClassifier(train);
            System.out.println("BUILT QUAD");
            System.out.println(" Optimal C ="+svmq.getC());
           svmrbf.buildClassifier(train);
            System.out.println("BUILT RBF");
            System.out.println(" Optimal C ="+svmrbf.getC());
            double accL=0,accQ=0,accRBF=0;
            accL=ClassifierTools.accuracy(test, svml);
           accQ=ClassifierTools.accuracy(test, svmq);
           accRBF=ClassifierTools.accuracy(test,svmrbf);

            System.out.println("ACC on "+problemFile+": Linear = "+df.format(accL)+", Quadratic = "+df.format(accQ)+", RBF = "+df.format(accRBF));
                
         }catch(Exception e){
            System.out.println(" Exception builing a classifier = "+e);
            System.exit(0);
        }
    }

    @Override
    public void setCVPath(String train) {
        trainPath=train;
    
    }

    @Override
    public String getParameters() {
        String result="CVAcc,"+cvAcc+",C,"+paras[0];
        if(paras.length>1)
            result+=",Gamma,"+paras[1];
        return result;
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
