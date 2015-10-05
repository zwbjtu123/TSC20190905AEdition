package tsc_algorithms;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class PS_Ensemble extends AbstractClassifier{
    Classifier baseClassifier;
    Instances format;
    ClassifierType c=ClassifierType.RandF;
    //Power Spectrum
    PowerSpectrum ps=new PowerSpectrum();
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
                
        } 
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        ps=new PowerSpectrum();
        baseClassifier=c.createClassifier();
        Instances psTrain=ps.process(data);
        baseClassifier.buildClassifier(psTrain);
        format=new Instances(data,0);
    }
    @Override
    public double classifyInstance(Instance ins) throws Exception{
//   
        format.add(ins);    //Should match!
        Instances temp=ps.process(format);
        Instance trans=temp.get(0);
        format.remove(0);
        return baseClassifier.classifyInstance(trans);
    }
}
