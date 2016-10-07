/*
Shaplet transform with the weighted ensemble
 */
package tsc_algorithms;

import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import java.io.File;
import java.math.BigInteger;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.timeseriesensembles.depreciated.HESCA_05_10_16;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;
import static weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory.nanoToOp;
import weka.filters.timeseries.shapelet_transforms.searchFuntions.ShapeletSearch;

/**
 *
 * @author raj09hxu
 */
public class ST_Ensemble  extends AbstractClassifier implements SaveableEnsemble{

    public enum ST_TimeLimit {MINUTE, HOUR, DAY};

    //Minimum number of instances per class in the train set
    public static final int minimumRepresentation = 25;
    
    
    private HESCA_05_10_16 hesca;
    private ShapeletTransform transform;
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
        String ensemble=hesca.getParameters();
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
    
    //set any value in nanoseconds you like.
    public void setTimeLimit(long time){
        timeLimit = time;
    }
    
    //pass in an enum of hour, minut, day, and the amount of them. 
    public void setTimeLimit(ST_TimeLimit time, int amount){
        //min,hour,day in longs.
        long[] times = {ShapeletTransformFactory.dayNano/24/60, ShapeletTransformFactory.dayNano/24, ShapeletTransformFactory.dayNano};
        
        timeLimit = times[time.ordinal()] * amount;
    }
    
    public void setOneDayLimit(){
        setTimeLimit(ST_TimeLimit.DAY, 1);
    }
    
    public void setOneHourLimit(){
        setTimeLimit(ST_TimeLimit.HOUR, 1);
    }
    
    public void setOneMinuteLimit(){
        setTimeLimit(ST_TimeLimit.MINUTE, 1);
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        format = doTransform ? createTransformData(data, timeLimit) : data;
        
        hesca=new HESCA_05_10_16();
        hesca.setWeightType("prop");
                
        redundantFeatures=InstanceTools.removeRedundantTrainAttributes(format);
        if(saveResults){
            hesca.setCVPath(trainCV);
            hesca.saveTestPreds(testPredictions);
        }
        
        System.out.println("transformed");
        hesca.buildClassifier(format);
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
        return hesca.classifyInstance(test);
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
        return hesca.distributionForInstance(test);
    }

    public Instances createTransformData(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;

        //construct shapelet classifiers from the factory.
        transform = ShapeletTransformFactory.createTransform(train);
        
        //Stop it printing everything
        transform.supressOutput();
        
        //at the moment this could be overrided.
        //transform.setSearchFunction(new LocalSearch(3, m, 10, seed));

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
            transform.setNumberOfShapelets(subsample.numInstances());
            transform.setSearchFunction(new ShapeletSearch(3, m, i, i));
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
