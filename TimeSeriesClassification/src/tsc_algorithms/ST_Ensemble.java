/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tsc_algorithms;

import bakeOffExperiments.Experiments;
import java.io.File;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory.calculateOperations;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory.opCountThreshold;

/**
 *
 * @author raj09hxu
 */
public class ST_Ensemble  extends AbstractClassifier implements SaveableEnsemble{

    private WeightedEnsemble weightedEnsemble;
    private FullShapeletTransform transform;
    private Instances format;
    int[] redundantFeatures;
    private boolean saveResults=false;
    private String trainCV="";
    private String testPredictions="";
    private boolean doTransform=true;
    
    protected void saveResults(boolean s){
        saveResults=s;
    }
        
    @Override
    public void saveResults(String tr, String te){
        saveResults(true);
        trainCV=tr;
        testPredictions=te;
    }
    
    public void doSTransform(boolean b){
        doTransform=b;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        format = doTransform ? createTransformData(data) : data;
        
        weightedEnsemble=new WeightedEnsemble();
        weightedEnsemble.setWeightType("prop");
                
        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(format);
        if(saveResults){
            weightedEnsemble.saveTrainCV(trainCV);
            weightedEnsemble.saveTestPreds(testPredictions);
        }
        
        weightedEnsemble.buildClassifier(format);
        format=new Instances(data,0);
    }
    
     @Override
    public double classifyInstance(Instance ins) throws Exception{
        format.add(ins);
        
        Instances temp  = doTransform ? transform.process(format) : format;
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        format.remove(0);
        return weightedEnsemble.classifyInstance(test);
    }

    public Instances createTransformData(Instances data){
        int numInstances = data.numInstances();
        int numAttributes = data.numAttributes()-1;
        
        //construct shapelet classifiers from the factory.
        transform = ShapeletTransformFactory.createTransform(data);
        
        //here we should check whether we need to subsample.
        //we should only subsample if our time cutoff is exceeded. 
        //TODO: this needs to be a bit smarter
        if(opCountThreshold <= calculateOperations(numInstances,numAttributes, 3, numAttributes)){
            System.out.println("using subsampling");
            double proportion = utilities.InstanceTools.calculateSubSampleProportion(data, 25);
            Instances subsample = utilities.InstanceTools.subSampleFixedProportion(data, proportion, new Random().nextInt());
            transform.process(subsample);
        }
        
        return transform.process(data);
    }
    
    public static void main(String[] args) {
        String dataLocation = "..\\..\\resampled transforms\\BalancedClassShapeletTransform\\";
        String saveLocation = "..\\..\\resampled results\\BalancedClassShapeletTransform\\";
        String datasetName = "ItalyPowerDemand";
        int fold = 0;
        
        Instances train= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+fold+"_TRAIN");
        Instances test= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+fold+"_TEST");
        String trainS= saveLocation+datasetName+File.separator+"TrainCV.csv";
        String testS=saveLocation+datasetName+File.separator+"TestPreds.csv";
        String preds=saveLocation+datasetName;

        ST_Ensemble st= new ST_Ensemble();
        st.saveResults(trainS, testS);
        st.doSTransform(true);
        double a = Experiments.singleSampleExperiment(train, test, st, fold, preds);
        System.out.println("accuracy: " + a);
    }    
}
