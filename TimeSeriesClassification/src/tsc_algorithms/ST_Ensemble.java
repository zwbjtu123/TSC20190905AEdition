/*
Shaplet transform with the weighted ensemble
 */
package tsc_algorithms;

import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import java.io.File;
import java.math.BigInteger;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory.nanoToOp;
import weka.filters.timeseries.shapelet_transforms.searchFuntions.LocalSearch;
import weka.filters.timeseries.shapelet_transforms.searchFuntions.ShapeletSearch;

/**
 *
 * @author raj09hxu
 */
public class ST_Ensemble  extends AbstractClassifier implements SaveableEnsemble{

    private static final int minimumRepresentation = 25;
    
    
    private WeightedEnsemble weightedEnsemble;
    private FullShapeletTransform transform;
    private Instances format;
    int[] redundantFeatures;
    private boolean saveResults=false;
    private String trainCV="";
    private String testPredictions="";
    private boolean doTransform=true;
    
    private long seed = 0;
    private long timeLimit = Long.MAX_VALUE;
    
    protected void saveResults(boolean s){
        saveResults=s;
    }
    
    public void setSeed(long sd){
        seed = sd;
    }
        
    @Override
    public void saveResults(String tr, String te){
        saveResults(true);
        trainCV=tr;
        testPredictions=te;
 //       transform.
    }
    @Override
    public String getParameters(){
        String paras=transform.getParameters();
        String ensemble=weightedEnsemble.getParameters();
        return paras+","+ensemble;
    }
    public void doSTransform(boolean b){
        doTransform=b;
    }
    
    
    public long getTransformOpCount(){
        return transform.getCount();
    }
    
    
    public Instances transformDataset(Instances data){
        if(transform.isFirstBatchDone())
            return transform.process(data);
        return null;
    }
    
    public void setTimeLimit(long time){
        timeLimit = time;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        format = doTransform ? createTransformData(data, timeLimit) : data;
        
        weightedEnsemble=new WeightedEnsemble();
        weightedEnsemble.setWeightType("prop");
                
        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(format);
        if(saveResults){
            weightedEnsemble.setCVPath(trainCV);
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
     @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        format.add(ins);
        
        Instances temp  = doTransform ? transform.process(format) : format;
//Delete redundant
        for(int del:redundantFeatures)
            temp.deleteAttributeAt(del);
        
        Instance test  = temp.get(0);
        format.remove(0);
        return weightedEnsemble.distributionForInstance(test);
    }

    public Instances createTransformData(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;
        
        //construct shapelet classifiers from the factory.
        transform = ShapeletTransformFactory.createTransform(train);
        transform.setSearchFunction(new LocalSearch(3, m, 10, seed));

        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        
        BigInteger opCount = ShapeletTransformFactory.calculateOps(n, m, 1, 1);
        
        //we need to resample.
        if(opCount.compareTo(opCountTarget) == 1){
            
            double recommendedProportion = ShapeletTransformFactory.calculateN(n, m, time);
            
            //calculate n for minimum class rep of 25.
            int small_sf = InstanceTools.findSmallestClassAmount(train);           
            double proportion = 1.0;
            if (small_sf>minimumRepresentation){
                proportion = (double)minimumRepresentation/(double)small_sf;
            }
            
            //if recommended is smaller than our cutoff threshold set it to the cutoff.
            if(recommendedProportion < proportion){
                recommendedProportion = proportion;
            }
            
            //subsample out dataset.
            Instances subsample = utilities.InstanceTools.subSampleFixedProportion(train, recommendedProportion, seed);
            
            int i=1;
            //if we've properly resampled this should pass on first go. IF we haven't we'll try and reach our target. 
            //calculate N is an approximation, so the subsample might need to tweak q and p just to bring us under. 
            while(ShapeletTransformFactory.calculateOps(subsample.numInstances(), m, i, i).compareTo(opCountTarget) == 1){
                i++;
            }
            
            System.out.println("new q and p: " + i);
            
            
            double percentageOfSeries = (double)i/(double)m * 100.0;
            
            System.out.println("percentageOfSeries: "+ percentageOfSeries);

            System.out.println("new n: " + subsample.numInstances());
            
            

            //we should look for less shapelets if we've resampled. 
            //e.g. Eletric devices can be sampled to from 8000 for 2000 so we should be looking for 20,000 shapelets not 80,000
            transform.setNumberOfShapelets(subsample.numInstances()*10);
            //transform.setSearchFunction(new ShapeletSearch(3, m, i, i));
            transform.process(subsample);
        }
        
        return transform.process(train);
    }
    
    public static void main(String[] args) throws Exception {
        String dataLocation = "C:\\LocalData\\Dropbox\\TSC Problems (1)\\";
        //String dataLocation = "..\\..\\resampled transforms\\BalancedClassShapeletTransform\\";
        String saveLocation = "..\\..\\resampled results\\BalancedClassShapeletTransform\\";
        String datasetName = "HeartbeatBIDMC";
        int fold = 0;
        
        Instances train= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+"_TRAIN");
        Instances test= ClassifierTools.loadData(dataLocation+datasetName+File.separator+datasetName+"_TEST");
        String trainS= saveLocation+datasetName+File.separator+"TrainCV.csv";
        String testS=saveLocation+datasetName+File.separator+"TestPreds.csv";
        String preds=saveLocation+datasetName;

        ST_Ensemble st= new ST_Ensemble();
        //st.saveResults(trainS, testS);
        st.doSTransform(true);
        st.setTimeLimit(ShapeletTransformFactory.dayNano);
        st.buildClassifier(train);
        double accuracy = utilities.ClassifierTools.accuracy(test, st);
        
        System.out.println("accuracy: " + accuracy);
    }    
}
