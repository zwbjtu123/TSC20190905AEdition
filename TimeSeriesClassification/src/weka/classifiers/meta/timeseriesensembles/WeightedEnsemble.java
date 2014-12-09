/*
 this classifier does none of the transformations. It simply loads the 
 * problems it is told to. In build classifier it can load the CV weights or find them, 
 * by default through LOOCV. For classifiers, it defaults to a standard set with default 
 * parameters. Alternatively, you can set the classifiers and for certain types set the
 * parameters through CV. 
 */
package weka.classifiers.meta.timeseriesensembles;

import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.*;
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
    double[] weights;
    boolean loadCVWeights=false;
    public static int MAX_NOS_FOLDS=500;
    enum WeightType{EQUAL,BEST,PROPORTIONAL,SIGNIFICANT_BINOMIAL,SIGNIFICANT_MCNEMAR};
    WeightType w;
    public WeightedEnsemble(){
        w=WeightType.PROPORTIONAL;
        classifierNames=new ArrayList<String>();
        c=setDefaultClassifiers(classifierNames);
        weights=new double[c.length];
    }
    public WeightedEnsemble(Classifier[] cl,ArrayList<String> names){
        w=WeightType.PROPORTIONAL;
        setClassifiers(cl,names);
        weights=new double[c.length];
    }
    public void setWeightType(WeightType a){w=a;}
    public void setWeightType(String s){
        String str=s.toUpperCase();
        w=WeightType.EQUAL;
/*        switch(str){
            case "EQUAL": case "EQ": case "E":
                w=WeightType.EQUAL;
                break;
            case "BEST": case "B":
                w=WeightType.BEST;
                break;
            case "PROPORTIONAL": case "PROP": case "P":
                w=WeightType.PROPORTIONAL;
                break;
            case "SIGNIFICANT_BINOMIAL": case "SIGB": case "SIG": case "S":
                w=WeightType.SIGNIFICANT_BINOMIAL;
                break;
            case "SIGNIFICANT_MCNEMAR": case "SIGM": case "SM":
                w=WeightType.SIGNIFICANT_MCNEMAR;
                break;
                
                
        }
  */  
    }

    //Default to kNN, Naive Bayes, C4.5, SVML, SVMQ, RandForest200, RotForest50
    final public Classifier[] setDefaultClassifiers(ArrayList<String> names){
            ArrayList<Classifier> sc2=new ArrayList<Classifier>();
            sc2.add(new kNN(1));
            names.add("NN");
            Classifier c;
            sc2.add(new NaiveBayes());
            names.add("NB");
            sc2.add(new J48());
            names.add("C45");
            c=new SMO();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            ((SMO)c).setKernel(kernel);
            sc2.add(c);
            names.add("SVML");
            c=new SMO();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            ((SMO)c).setKernel(kernel);
            sc2.add(c);
            names.add("SVMQ");
            c=new RandomForest();
            ((RandomForest)c).setNumTrees(100);
            sc2.add(c);
            names.add("RandF200");
            c=new RotationForest();
            sc2.add(c);
            names.add("RotF50");

            Classifier[] sc=new Classifier[sc2.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=sc2.get(i);

            return sc;
    }


    final public void setClassifiers(Classifier[] cl,  ArrayList<String> names){
        c=cl;
        classifierNames=new ArrayList<String>(names);
    }

    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        train = data;
//HERE: Offer the option of loading CV, classifiers and parameter sets        
//NOT IMPLEMENTED YET
        
//Parameter optimisation first, 
//NOT IMPLEMENTED YET

//Train the classifiers        
        for(int i=0;i<c.length;i++)
            c[i].buildClassifier(train);
//Find the weights of the classifier through CV
        if(w!=WeightType.EQUAL){
            Evaluation eval=new Evaluation(train);
            for(int i=0;i<c.length;i++){
                Random r= new Random();
                r.setSeed(1234);
//Assume LOOCV, but set the max number of folds to 500
                int folds=train.numInstances();
                if(folds>MAX_NOS_FOLDS)
                    folds=MAX_NOS_FOLDS;
                eval.crossValidateModel(c[i],train,folds,r);
                weights[i]=1-eval.errorRate();
            }
        }
        else{
            for(int i=0;i<c.length;i++)
                weights[i]=1/(double)c.length;
        }
        if(w==WeightType.BEST){ //Find largest, set to one and others to zero. 
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
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        double[] preds=new double[ins.numClasses()];
        for(int i=0;i<c.length;i++){
            int p=(int)c[i].classifyInstance(ins);
//            System.out.println(" Classifier "+classifierNames.get(i)+" predicts class "+p+" with weight "+weights[i]);
            preds[p]+=weights[i];
        }
        double sum=preds[0];
        for(int i=1;i<preds.length;i++)
            sum+=preds[i];
        for(int i=0;i<preds.length;i++)
            preds[i]/=sum;
        
        return preds;
    }
    public ArrayList<String> getNames(){ return classifierNames;}
//Test for the ensemble in the spectral data
 /*   public static void testSpectrum() throws Exception{
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PSItalyPowerDemand\\PSItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PSItalyPowerDemand\\PSItalyPowerDemand_TEST");
        WeightedEnsemble w=new WeightedEnsemble();
        w.buildClassifier(train);
        System.out.println(" Accuracy ="+ClassifierTools.accuracy(test, w));
    }
*/    public static void main(String[] args){
        try{
//            testSpectrum();
        }catch(Exception e){
            
        }
    }
}
