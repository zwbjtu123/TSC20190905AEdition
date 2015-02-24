/*
 this classifier does none of the transformations. It simply loads the 
 * problems it is told to. In build classifier it can load the CV weights or find them, 
 * by default through LOOCV. For classifiers, it defaults to a standard set with default 
 * parameters. Alternatively, you can set the classifiers and for certain types set the
 * parameters through CV. 
 */
package weka.classifiers.meta.timeseriesensembles;

import development.DataSets;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.*;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

/**
 *
 * @author ajb
 */
public class WeightedEnsemble extends AbstractClassifier{
//The McNemar test requires the actual predictions of each classifier. The others can be found directly
//from the CV accuracy.    
    Instances train;
    Classifier[] c;
    ArrayList<String> classifierNames;
    double[] cvAccs;
    double[] weights;
    boolean loadCVWeights=false;
    public static int MAX_NOS_FOLDS=50;
    public enum WeightType{EQUAL,BEST,PROPORTIONAL,SIGNIFICANT_BINOMIAL,SIGNIFICANT_MCNEMAR};
    WeightType w;
    public WeightedEnsemble(){
        w=WeightType.PROPORTIONAL;
        classifierNames=new ArrayList<String>();
        c=setDefaultClassifiers(classifierNames);
        weights=new double[c.length];
        cvAccs=new double[c.length];
    }
    public WeightedEnsemble(Classifier[] cl,ArrayList<String> names){
        w=WeightType.PROPORTIONAL;
        setClassifiers(cl,names);
        weights=new double[c.length];
        cvAccs=new double[c.length];
    }
    public void setWeightType(WeightType a){w=a;}
    public void setWeightType(String s){
        String str=s.toUpperCase();
        switch(str){
            case "EQUAL": case "EQ": case "E":
                w=WeightType.EQUAL;
                break;
            case "BEST": case "B":
                w=WeightType.BEST;
                break;
            case "PROPORTIONAL": case "PROP": case "P":
                w=WeightType.PROPORTIONAL;
                break;
 /*            case "SIGNIFICANT_BINOMIAL": case "SIGB": case "SIG": case "S":
                w=WeightType.SIGNIFICANT_BINOMIAL;
                break;
            case "SIGNIFICANT_MCNEMAR": case "SIGM": case "SM":
                w=WeightType.SIGNIFICANT_MCNEMAR;
                break;
 */      
            default:
                throw new UnsupportedOperationException("Weighting method "+str+" not supported yet.");       
        }
   
    }

/*The classifiers used are the WEKA [26] implementations
of k Nearest Neighbour (where k is set through cross
validation), Naive Bayes, C4.5 decision tree [27], Support
Vector Machines [28] with linear and quadratic basis
function kernels, Random Forest [29] (with 100 trees), Ro-
tation Forest [30] (with 10 trees), and a Bayesian network.
 */       
    final public Classifier[] setDefaultClassifiers(ArrayList<String> names){
            ArrayList<Classifier> classifiers=new ArrayList<>();
            kNN k=new kNN(100);
            k.setCrossValidate(true);
            k.normalise(false);
            k.setDistanceFunction(new EuclideanDistance());
            classifiers.add(k);
            names.add("NN");
            classifiers.add(new NaiveBayes());
            names.add("NB");
            classifiers.add(new J48());
            names.add("C45");
            SMO svm=new SMO();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            svm.setKernel(kernel);
            classifiers.add(svm);
            names.add("SVML");
            svm=new SMO();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            svm.setKernel(kernel);
            classifiers.add(svm);
            names.add("SVMQ");
            RandomForest r=new RandomForest();
            r.setNumTrees(100);
            classifiers.add(r);
            names.add("RandF100");
            RotationForest rf=new RotationForest();
            rf.setNumIterations(30);
            classifiers.add(rf);
            names.add("RotF30");
            BayesNet bn=new BayesNet();
            classifiers.add(bn);
            names.add("bayesNet");
            
            
            Classifier[] sc=new Classifier[classifiers.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=classifiers.get(i);

            return sc;
    }


    final public void setClassifiers(Classifier[] cl,  ArrayList<String> names){
        c=cl;
        classifierNames=new ArrayList<>(names);
        weights=new double[c.length];
        cvAccs=new double[c.length];
    }

    public final double[] getWeights(){ return weights;}
    public final double[] getCVAccs(){ return cvAccs;}
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        train = data;
//HERE: Offer the option of loading CV, classifiers and parameter sets        
//NOT IMPLEMENTED YET
        
//Parameter optimisation first, 
//NOT IMPLEMENTED YET

/*Train the classifiers on the whole train data. These are the classifiers        
        that will be used for predictions
 */
        for(int i=0;i<c.length;i++)
            c[i].buildClassifier(train);
        
//If using equal weighting, set cvAcc to 1 and weights to 1/nosClassifiers        
        if(w==WeightType.EQUAL){
            for(int i=0;i<c.length;i++){
                cvAccs[i]=1;
                weights[i]=1/(double)c.length;
            }
        }
//Else, find the cvAccs of the classifier through CV, then weight proportionally
//All weight types will require this        
        else{
            double sum=0;
            for(int i=0;i<c.length;i++){
                Evaluation eval=new Evaluation(train);
                Random r= new Random();
//                r.setSeed(1234);
//set the max number of folds to 100 or use LOOCV
                int folds=train.numInstances();
                if(folds>MAX_NOS_FOLDS)
                    folds=MAX_NOS_FOLDS;
                eval.crossValidateModel(c[i],train,folds,r);
                cvAccs[i]=1-eval.errorRate();
                sum+=cvAccs[i];
            }
            for(int i=0;i<c.length;i++)
                weights[i]=cvAccs[i]/sum;
        }
        
 //If using Best find largest, set to one and others to zero.       
        if(w==WeightType.BEST){  
            int bestPos=0;
            for(int i=1;i<weights.length;i++){
                if(weights[i]>weights[bestPos])
                    bestPos=i;
            }
            for(int i=0;i<weights.length;i++){
                if(i==bestPos)
                    weights[i]=1;
                else
                    weights[i]=0;
            }
        }
    }
    
//Stores the current individual classifier predictions for the last call to
//classifyInstance
    private double[] predictions;
    
    public double[] getPredictions(){ return predictions;}
    
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        predictions=new double[c.length];
        double[] preds=new double[ins.numClasses()];
        for(int i=0;i<c.length;i++){
            predictions[i]=c[i].classifyInstance(ins);
//            System.out.println(" Classifier "+classifierNames.get(i)+" predicts class "+predictions[i]+" with weight "+weights[i]);
            preds[(int)predictions[i]]+=weights[i];
        }
        double sum=preds[0];
        for(int i=1;i<preds.length;i++)
            sum+=preds[i];
        for(int i=0;i<preds.length;i++)
            preds[i]/=sum;
        
        return preds;
    }
    public ArrayList<String> getNames(){ return classifierNames;}

    
    public static void debugTest(){
//Basic test of build classifer 
        String problem="ItalyPowerDemand";
        Instances train =ClassifierTools.loadData(DataSets.dropboxPath+problem+"\\"+problem+"_TRAIN");
        Instances test =ClassifierTools.loadData(DataSets.dropboxPath+problem+"\\"+problem+"_TEST");
        WeightedEnsemble we = new WeightedEnsemble();
//Test equal weight and CV weight for small classifier set
        DecimalFormat df = new DecimalFormat("###.###");
        double a;
        try {
            Classifier[] c = new Classifier[3];
            ArrayList<String> names = new ArrayList<>();
            kNN k=new kNN(100);
            k.setCrossValidate(true);
            k.normalise(false);
            k.setDistanceFunction(new EuclideanDistance());
            c[0]=k;
            names.add("NN");
            c[1]=new NaiveBayes();
            names.add("NB");
            c[2]=new J48();
            names.add("C45");
            we.setClassifiers(c, names);
            we.setWeightType("EQUAL");
            we.buildClassifier(train);
            a=ClassifierTools.accuracy(test, we);
            System.out.println(" WE accuracy with equal weight="+a);
            we.setWeightType("Proportional");
            we.buildClassifier(train);
            double[] w=we.getWeights();
            double[] cv=we.getCVAccs();
            for(int i=0;i<w.length;i++)
                System.out.println("Weight ="+df.format(w[i])+" CV ="+df.format(cv[i]));
            a=ClassifierTools.accuracy(test, we);
            System.out.println(" WE accuracy with prop weight="+a);
            
        } catch (Exception ex) {
            Logger.getLogger(WeightedEnsemble.class.getName()).log(Level.SEVERE, null, ex);
        }
//Test with standard classifiers
        try {
            we = new WeightedEnsemble();
            we.setWeightType("Proportional");
            we.buildClassifier(train);
            double[] w=we.getWeights();
            double[] cv=we.getCVAccs();
            for(int i=0;i<w.length;i++)
                System.out.println("Weight ="+df.format(w[i])+" CV ="+df.format(cv[i]));
            a=ClassifierTools.accuracy(test, we);
            System.out.println(" WE accuracy with prop weight="+a);
            
            
        } catch (Exception ex) {
            Logger.getLogger(WeightedEnsemble.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
    }
//Test for the ensemble in the spectral data
 /*   public static void testSpectrum() throws Exception{
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PSItalyPowerDemand\\PSItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PSItalyPowerDemand\\PSItalyPowerDemand_TEST");
        WeightedEnsemble w=new WeightedEnsemble();
        w.buildClassifier(train);
        System.out.println(" Accuracy ="+ClassifierTools.accuracy(test, w));
    }
*/    public static void main(String[] args){
            debugTest();
    }
}
