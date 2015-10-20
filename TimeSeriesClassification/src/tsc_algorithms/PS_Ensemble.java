package tsc_algorithms;

import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class PS_Ensemble extends AbstractClassifier implements SaveableEnsemble{
    Classifier baseClassifier;
    Instances format;
    ClassifierType c=ClassifierType.RandF;
    private boolean saveResults=false;
    private String trainCV="";
    private String testPredictions="";
    int[] constantFeatures;
    PowerSpectrum ps=new PowerSpectrum();
    boolean doTransform=true;
    public PowerSpectrum getTransform(){ return ps;}
    protected void saveResults(boolean s){
        saveResults=s;
    }
    public void saveResults(String tr, String te){
        saveResults(true);
        trainCV=tr;
        testPredictions=te;
    }
     public void doTransform(boolean b){
        doTransform=b;
    }
   
    //Power Spectrum
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
                case "WE":
                   WeightedEnsemble we=new WeightedEnsemble();
                   we.setWeightType("prop");
                   return we;
                default:
                   RandomForest c=new RandomForest();
                   c.setNumTrees(numBaseClassifiers);
                    return c;
            }
        }
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
            case "weightedensemble": case "we": case "wens": 
                c=ClassifierType.WeightedEnsemble;
                break;                
        } 
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        ps=new PowerSpectrum();
        baseClassifier=c.createClassifier();
        Instances psTrain;
        if(doTransform)
             psTrain=ps.process(data);
        else
             psTrain=data;
        constantFeatures=InstanceTools.removeConstantTrainAttributes(psTrain);

        if(saveResults && c==ClassifierType.WeightedEnsemble){
//Set up the file space here
            ((WeightedEnsemble) baseClassifier).saveTrainCV(trainCV);
            ((WeightedEnsemble) baseClassifier).saveTestPreds(testPredictions);
        }
        
        baseClassifier.buildClassifier(psTrain);
        format=new Instances(data,0);
    }
    @Override
    public double classifyInstance(Instance ins) throws Exception{
//   
        format.add(ins);    //Should match!
        Instances temp;
        if(doTransform)
            temp=ps.process(format);
        else
            temp=format;
//Delete constants
        for(int del:constantFeatures)
            temp.deleteAttributeAt(del);
        Instance trans=temp.get(0);
        format.remove(0);
        return baseClassifier.classifyInstance(trans);
    }
}
