/*
Shaplet transform with the weighted ensemble
 */
package tsc_algorithms.depreciated;

import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import java.io.File;
import java.math.BigInteger;
import shapelet_transforms.ShapeletTransform;
import shapelet_transforms.ShapeletTransformFactory;
import shapelet_transforms.ShapeletTransformFactoryOptions;
import shapelet_transforms.ShapeletTransformTimingUtilities;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.timeseriesensembles.depreciated.HESCA_05_10_16;
import weka.core.Instance;
import weka.core.Instances;
import static shapelet_transforms.ShapeletTransformTimingUtilities.nanoToOp;
import shapelet_transforms.distance_functions.SubSeqDistance;
import shapelet_transforms.quality_measures.ShapeletQuality;
import shapelet_transforms.search_functions.ShapeletSearch;
import shapelet_transforms.search_functions.ShapeletSearchOptions;

/**
 *
 * @author raj09hxu
 */
@Deprecated
public class ST_HESCA  extends AbstractClassifier implements SaveableEnsemble{

    public enum ST_TimeLimit {MINUTE, HOUR, DAY};

    //Minimum number of instances per class in the train set
    public static final int minimumRepresentation = 25;
    
    private String shapeletOutputPath;
    
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
    
    public ShapeletTransform getTransform(){
        return transform;
    }
    
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
        long[] times = {ShapeletTransformTimingUtilities.dayNano/24/60, ShapeletTransformTimingUtilities.dayNano/24, ShapeletTransformTimingUtilities.dayNano};
        
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
        
//        System.out.println("transformed");
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
    
    public void setShapeletOutputFilePath(String path){
        shapeletOutputPath = path;
    }

    public Instances createTransformData(Instances train, long time){
        int n = train.numInstances();
        int m = train.numAttributes()-1;

        //construct the options for the transform.
        ShapeletTransformFactoryOptions.Builder optionsBuilder = new ShapeletTransformFactoryOptions.Builder();
        optionsBuilder.setDistanceType(SubSeqDistance.DistanceType.IMP_ONLINE);
        optionsBuilder.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);
        if(train.numClasses() > 2){
            optionsBuilder.useBinaryClassValue();
            optionsBuilder.useClassBalancing();
        }
        optionsBuilder.useRoundRobin();
        optionsBuilder.useCandidatePruning();
        optionsBuilder.setKShapelets(train.numInstances());
        
        //we use a 
        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setMin(3);
        searchBuilder.setMax(m);
        searchBuilder.setSearchType(ShapeletSearch.SearchType.FULL);
        

        //at the moment this could be overrided.
        BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
        
        BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
        
        //we need to resample.
        if(opCount.compareTo(opCountTarget) == 1){
            double recommendedProportion = ShapeletTransformTimingUtilities.calculateN(n, m, time);
            
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
            while(ShapeletTransformTimingUtilities.calculateOps(subsample.numInstances(), m, i, i).compareTo(opCountTarget) == 1){
                i++;
            }
            //add extra options for the 
            searchBuilder.setLengthInc(i);
            searchBuilder.setPosInc(i);
            optionsBuilder.setKShapelets(subsample.numInstances());
            optionsBuilder.setSearchOptions(searchBuilder.build());
            transform = new ShapeletTransformFactory(optionsBuilder.build()).getTransform();
            
            transform.supressOutput();
            if(shapeletOutputPath != null)
                transform.setLogOutputFile(shapeletOutputPath);
            
            //build shapelet set from subsample.
            transform.process(subsample);
        }else{
            optionsBuilder.setSearchOptions(searchBuilder.build());
            transform = new ShapeletTransformFactory(optionsBuilder.build()).getTransform();
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

        ST_HESCA st= new ST_HESCA();
        //st.saveResults(trainS, testS);
        st.doSTransform(true);
        st.setTimeLimit(ShapeletTransformTimingUtilities.dayNano);
        st.buildClassifier(train);
        double accuracy = utilities.ClassifierTools.accuracy(test, st);
        
        System.out.println("accuracy: " + accuracy);
    }    
}
