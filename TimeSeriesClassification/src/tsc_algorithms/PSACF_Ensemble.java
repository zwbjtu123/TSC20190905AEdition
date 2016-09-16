package tsc_algorithms;


import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import fileIO.OutFile;
import java.util.ArrayList;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
easiest way to generate these is to deconstruct the weighted ensemble.  
 */
public class PSACF_Ensemble extends AbstractClassifier implements SaveableEnsemble{

    public enum ClassifierType{
        RandF("RandF",500),RotF("RotF",50),WeightedEnsemble("WE",8);
        String type;
        int numBaseClassifiers;
        ClassifierType(String s, int x){
            type=s;
            numBaseClassifiers=x;
        }
        Classifier createClassifier(){
            switch(type){
                case "RandF":
                   RandomForest randf=new RandomForest();
                   randf.setNumTrees(numBaseClassifiers);
                    return randf;
                case "RotF":
                   RotationForest rotf=new RotationForest();
                   rotf.setNumIterations(numBaseClassifiers);
                   return rotf;
                case "WE": case "HESC":
                   HESCA we=new HESCA();
                   we.setWeightType("prop");
                   return we;
                default:
                   RandomForest c=new RandomForest();
                   c.setNumTrees(numBaseClassifiers);
                    return c;
            }
        }
    }
    public enum TransformType{ACF,PS,PSACF};

    private ClassifierType c=ClassifierType.WeightedEnsemble;
    private TransformType transform=TransformType.ACF;
    PowerSpectrum ps=new PowerSpectrum();
    private Classifier baseClassifier;
    private Instances format;
    private boolean saveResults=false;
    private String trainCV="";
    private String testPredictions="";
    private boolean doTransform=true;
    int[] constantFeatures;
    
    public void setTransformType(String str){
        str=str.toUpperCase();
        switch(str){
            case "ACF": case "AR":
                transform=TransformType.ACF;
                break;
            case "PS": case "POWERSPECTRUM":
                transform=TransformType.PS;
                break;
            case "PSACF": case "ACFPS": case "BOTH":
                transform=TransformType.PSACF;
                break;
        }
    }
    protected void saveResults(boolean s){
        saveResults=s;
    }
    public void saveResults(String tr, String te){
        saveResults(true);
        trainCV=tr;
        testPredictions=te;
    }

    @Override
    public String getParameters() {
        String str =c.toString()+","+transform.toString();
        return str;
    }
    
    
    public  void setClassifierType(String s){
        s=s.toLowerCase();
        switch(s){
            case "randf": case "randomforest": case "randomf":
                c=ClassifierType.RandF;
                break;
            case "rotf": case "rotationforest": case "rotationf":
                c=ClassifierType.RotF;
                break;
            case "weightedensemble": case "we": case "wens": case "hesca":
                c=ClassifierType.WeightedEnsemble;
                break;
                
        } 
    }
    public void doTransform(boolean b){
        doTransform=b;
    }
   
    private Instances combinedPSACF(Instances data)throws Exception {
        Instances combo=ACF.formChangeCombo(data);
        Instances temp2=ps.process(data);
        combo.setClassIndex(-1);
        combo.deleteAttributeAt(combo.numAttributes()-1); 
        combo=Instances.mergeInstances(combo, temp2);
        combo.setClassIndex(combo.numAttributes()-1);
        return combo;        

    }
    @Override
    public void buildClassifier(Instances data) throws Exception {
        baseClassifier=c.createClassifier();
/*This flag allows the perfomance of the transform outside of the classifier, 
        which speeds up the cross validation and is ok because the transform
        is unsupervised and indendent of other cases
        */
        Instances train;
        if(doTransform){
            switch(transform){
                case ACF:
                    train=ACF.formChangeCombo(data);
                    break;
                case PS: 
                    train=ps.process(data);
                    break;
                case PSACF:
                    train=combinedPSACF(data);
                    break;
                default:
                    train=ACF.formChangeCombo(data);
                    break;
            }
        }
        else
            train=data;
        constantFeatures=InstanceTools.removeConstantTrainAttributes(train);
        System.out.println("Number of constant features ="+constantFeatures.length);
        if(saveResults && c==ClassifierType.WeightedEnsemble){
//Set up the file space here
            ((HESCA) baseClassifier).setCVPath(trainCV);
            ((HESCA) baseClassifier).saveTestPreds(testPredictions);
        }
        baseClassifier.buildClassifier(train);
        
//Record original format, empty of instances 
        format=new Instances(data,0);
    }
    private Instances doTransform(Instance ins) throws Exception{
        Instances temp;
        switch(transform){
            case ACF:
                temp=ACF.formChangeCombo(format);
                break;
            case PS: 
                temp=ps.process(format);
                break;
            case PSACF:
                temp=combinedPSACF(format);
                break;
            default:
                temp=ACF.formChangeCombo(format);
        }
        return temp;
    }
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        format.add(ins);    //Should match!
        Instances temp;
        if(doTransform){
            temp=doTransform(ins);
        }
        else
            temp=format;
//Delete constants
        for(int del:constantFeatures)
            temp.deleteAttributeAt(del);
        Instance trans=temp.get(0);
        format.remove(0);
        return baseClassifier.distributionForInstance(trans);
    
    }
    
    @Override
    public double classifyInstance(Instance ins) throws Exception{
//   
        format.add(ins);    //Should match!
        Instances temp;
        if(doTransform){
            temp=doTransform(ins);
        }
        else
            temp=format;
//Delete constants
        for(int del:constantFeatures)
            temp.deleteAttributeAt(del);
        Instance trans=temp.get(0);
        format.remove(0);
        return baseClassifier.classifyInstance(trans);
    }
    public static void main(String[] args) {
        /*        Instances train= ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test= ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        String trainS="C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\TestPreds.csv";
        String testS="C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\TrainCV.csv";
        String preds="C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand";
        PSACF_Ensemble acf= new PSACF_Ensemble();
        acf.setClassifierType("WE");
        acf.saveResults(trainS, testS);
        double a =BakeoffExperiments.singleSampleExperiment(train, test, acf,0, preds);*/
    }
}
