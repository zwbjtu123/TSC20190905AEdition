/*
 Shapelet Factory: determines shapelet type and parameters based on the input
 * 1. distance caching: used if there is enough memory. 
 *           Distance caching requires O(nm^2)  
 * 2. number of shapelets:  
 *           Set to n*m/10, with a max size max(1000,2*m,2*n) . Assume we will post process cluster
 * 3. shapelet length range: 3 to train.numAttributes()-1
 * 
 */
package weka.filters.timeseries.shapelet_transforms;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.Pair;
import weka.core.Instances;
import weka.core.shapelet.*;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.classValue.NormalClassValue;
import weka.filters.timeseries.shapelet_transforms.searchFuntions.ShapeletSearch;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.CachedSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.OnlineSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.SubSeqDistance;

/**
 *
 * @author Aaron Bostrom
 */
public class ShapeletTransformFactory
{
    //this is an arbritary cutoff value for whether we should start subsampling. It's about 7 days(ish). TODO: test.
    public static final long opCountThreshold = 1000000000000000l; 
    
    public static final long dayNano = 86400000000000l;
    public static final long nanoToOp = 10l; //1 op takes 10 nanoseconds.
    
    
    
    //we create the Map using params jon found.
    //lazy way to avoid reading a text file. 
    //it is immutable.
    public static final Map<String, Pair<Integer, Integer>> shapeletParams;
    static{
        shapeletParams = new HashMap<>();
        shapeletParams.put("Adiac", new Pair(3,10));
        shapeletParams.put("ArrowHead", new Pair(17,90));
        shapeletParams.put("Beef", new Pair(8,30));
        shapeletParams.put("BeetleFly", new Pair(30,101));
        shapeletParams.put("BirdChicken", new Pair(30,101));
        shapeletParams.put("Car", new Pair(16,57));
        shapeletParams.put("CBF", new Pair(46,90));
        shapeletParams.put("ChlorineConcentration", new Pair(7,20));
        shapeletParams.put("CinCECGtorso", new Pair(697,814));
        shapeletParams.put("Coffee", new Pair(18,30));
        shapeletParams.put("Computers", new Pair(15,267));
        shapeletParams.put("CricketX", new Pair(120,255));
        shapeletParams.put("CricketY", new Pair(132,262));
        shapeletParams.put("CricketZ", new Pair(118,257));
        shapeletParams.put("DiatomSizeReduction", new Pair(7,16));
        shapeletParams.put("DistalPhalanxOutlineAgeGroup", new Pair(7,31));
        shapeletParams.put("DistalPhalanxOutlineCorrect", new Pair(6,16));
        shapeletParams.put("DistalPhalanxTW", new Pair(17,31));
        shapeletParams.put("Earthquakes", new Pair(24,112));
        shapeletParams.put("ECGFiveDays", new Pair(24,76));
        shapeletParams.put("FaceAll", new Pair(70,128));
        shapeletParams.put("FaceFour", new Pair(20,120));
        shapeletParams.put("FacesUCR", new Pair(47,128));
        shapeletParams.put("Fiftywords", new Pair(170,247));
        shapeletParams.put("Fish", new Pair(22,60));
        shapeletParams.put("FordA", new Pair(50,298));
        shapeletParams.put("FordB", new Pair(38,212));
        shapeletParams.put("GunPoint", new Pair(24,55));
        shapeletParams.put("Haptics", new Pair(21,103));
        shapeletParams.put("Herrings", new Pair(30,101));
        shapeletParams.put("InlineSkate", new Pair(750,896));
        shapeletParams.put("ItalyPowerDemand", new Pair(7,14));
        shapeletParams.put("LargeKitchenAppliances", new Pair(13,374));
        shapeletParams.put("Lightning2", new Pair(47,160));
        shapeletParams.put("Lightning7", new Pair(20,80));
        shapeletParams.put("Mallat", new Pair(52,154));
        shapeletParams.put("MedicalImages", new Pair(9,35));
        shapeletParams.put("MiddlePhalanxOutlineAgeGroup", new Pair(8,31));
        shapeletParams.put("MiddlePhalanxOutlineCorrect", new Pair(5,12));
        shapeletParams.put("MiddlePhalanxTW", new Pair(7,31));
        shapeletParams.put("MoteStrain", new Pair(16,31));
        shapeletParams.put("NonInvasiveFatalECGThorax1", new Pair(5,61));
        shapeletParams.put("NonInvasiveFatalECGThorax2", new Pair(12,58));
        shapeletParams.put("OliveOil", new Pair(8,27));
        shapeletParams.put("OSULeaf", new Pair(141,330));
        shapeletParams.put("PhalangesOutlinesCorrect", new Pair(5,14));
        shapeletParams.put("Plane", new Pair(18,109));
        shapeletParams.put("ProximalPhalanxOutlineAgeGroup", new Pair(7,31));
        shapeletParams.put("ProximalPhalanxOutlineCorrect", new Pair(5,12));
        shapeletParams.put("ProximalPhalanxTW", new Pair(9,31));
        shapeletParams.put("PtNDeviceGroups", new Pair(51,261));
        shapeletParams.put("PtNDevices", new Pair(100,310));
        shapeletParams.put("RefrigerationDevices", new Pair(13,65));
        shapeletParams.put("ScreenType", new Pair(11,131));
        shapeletParams.put("ShapeletSim", new Pair(25,35));
        shapeletParams.put("SmallKitchenAppliances", new Pair(31,443));
        shapeletParams.put("SonyAIBORobotSurface1", new Pair(15,36));
        shapeletParams.put("SonyAIBORobotSurface2", new Pair(22,57));
        shapeletParams.put("StarlightCurves", new Pair(68,650));
        shapeletParams.put("SwedishLeaf", new Pair(11,45));
        shapeletParams.put("Symbols", new Pair(52,155));
        shapeletParams.put("SyntheticControl", new Pair(20,56));
        shapeletParams.put("ToeSegmentation1", new Pair(39,153));
        shapeletParams.put("ToeSegmentation2", new Pair(100,248));
        shapeletParams.put("Trace", new Pair(62,232));
        shapeletParams.put("TwoLeadECG", new Pair(7,13));
        shapeletParams.put("TwoPatterns", new Pair(20,71));
        shapeletParams.put("UWaveGestureLibraryX", new Pair(113,263));
        shapeletParams.put("UWaveGestureLibraryY", new Pair(122,273));
        shapeletParams.put("UWaveGestureLibraryZ", new Pair(135,238));
        shapeletParams.put("Wafer", new Pair(29,152));
        shapeletParams.put("WordSynonyms", new Pair(137,238));
        shapeletParams.put("Worms", new Pair(93,382));
        shapeletParams.put("WormsTwoClass", new Pair(46,377));
        shapeletParams.put("Yoga", new Pair(12,132));
        Collections.unmodifiableMap(shapeletParams);
    }
    
    public static final double MEM_CUTOFF = 0.5;
    public static final int MAX_NOS_SHAPELETS = 1000;    
    
    public static FullShapeletTransform createCachedTransform()
    {
        FullShapeletTransform st = new FullShapeletTransform();
        st.setSubSeqDistance(new CachedSubSeqDistance());
        return st;
    }
    
    public static FullShapeletTransform createOnlineTransform()
    {
        FullShapeletTransform st = new FullShapeletTransform();
        st.setSubSeqDistance(new OnlineSubSeqDistance());
        return st;
    }
    
    public static FullShapeletTransform createBasicTransform(int n, int m){
        FullShapeletTransform fst = new FullShapeletTransform();
        fst.setNumberOfShapelets(n * 10);
        fst.setShapeletMinAndMax(3, m);
        fst.supressOutput();
        return fst;
    }
    
    public static FullShapeletTransform createTransform(Instances train){
        int numClasses = train.numClasses();
        int numInstances = train.numInstances();
        int numAttributes = train.numAttributes()-1;
        
        FullShapeletTransform transform;
        if(numClasses == 2){
            transform = new FullShapeletTransform();
        }else{
            transform = new BalancedClassShapeletTransform();
            transform.setClassValue(new BinarisedClassValue());
        }
        
        //transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
        transform.setShapeletMinAndMax(3, numAttributes);
        transform.setNumberOfShapelets(numInstances*10);
        transform.useCandidatePruning();
        transform.turnOffLog();
        
        return transform;
    }
    
    
    long getAvailableMemory()
    {
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        long usedMemory = totalMemory - freeMemory;
        long availableMemory = maxMemory - usedMemory;
        return availableMemory;
    }

    // Method to estimate min/max shapelet length for a given data
    public static int[] estimateMinAndMax(Instances data, FullShapeletTransform st)
    {
        FullShapeletTransform st1 = null;
        try {
            //we need to clone the FST.
            st1 = st.getClass().newInstance();
            st1.setClassValue(st.classValue.getClass().newInstance());
            st1.setSubSeqDistance(st.subseqDistance.getClass().newInstance());
        } catch (InstantiationException | IllegalAccessException ex) {
            System.out.println("Exception: ");
        }
        
        if(st1 == null)
            st1 = new FullShapeletTransform();
        
        st1.supressOutput();
        
        ArrayList<Shapelet> shapelets = new ArrayList<>();
        st.supressOutput();
        st.turnOffLog();

        Instances randData = new Instances(data);
        Instances randSubset;

        for (int i = 0; i < 10; i++)
        {
            randData.randomize(new Random());
            randSubset = new Instances(randData, 0, 10);
            shapelets.addAll(st1.findBestKShapeletsCache(10, randSubset, 3, randSubset.numAttributes() - 1));
        }

        Collections.sort(shapelets, new ShapeletLengthComparator());
        int min = shapelets.get(24).getContent().length;
        int max = shapelets.get(74).getContent().length;

        return new int[]{min,max};
    }
    
    //bog standard min max estimation.
    public static int[] estimateMinAndMax(Instances data)
    {
        return estimateMinAndMax(data, new FullShapeletTransform());
    }
    
    //Class implementing comparator which compares shapelets according to their length
    public static class ShapeletLengthComparator implements Comparator<Shapelet>{
   
        @Override
        public int compare(Shapelet shapelet1, Shapelet shapelet2){
            int shapelet1Length = shapelet1.getContent().length;        
            int shapelet2Length = shapelet2.getContent().length;

            return Integer.compare(shapelet1Length, shapelet2Length);  
        }
    }
    
    public static long calculateNumberOfShapelets(Instances train, int minShapeletLength, int maxShapeletLength){      
        return calculateNumberOfShapelets(train.numInstances(), train.numAttributes()-1, minShapeletLength, maxShapeletLength);
    }
    
    //Aaron
    //verified on Trace dataset from Ye2011 with 7,480,200 shapelets : page 158.
    //we assume as fixed length.
    public static long calculateNumberOfShapelets(int numInstances, int numAttributes, int minShapeletLength, int maxShapeletLength){
        long numShapelets=0;
        
        //calculate number of shapelets in a single instance.
        for (int length = minShapeletLength; length <= maxShapeletLength; length++) {
            numShapelets += numAttributes - length + 1;
        }
        
        numShapelets*=numInstances;
        
        return numShapelets;
    }
    
    
    public static long calculateOperations(Instances train, int minShapeletLength, int maxShapeletLength){      
        return calculateOperations(train.numInstances(), train.numAttributes()-1, minShapeletLength, maxShapeletLength);
    }
    
    //verified correct by counting ops in transform
    public static long calculateOperations(int numInstances, int numAttributes, int minShapeletLength, int maxShapeletLength){
        return calculateOperationsWithSkipping(numInstances, numAttributes, minShapeletLength, maxShapeletLength, 1,1);
    }
    
    
     //verified correct by counting ops in transform
    public static long calculateOperationsWithSkipping(int numInstances, int numAttributes, int minShapeletLength, int maxShapeletLength, int posSkip, int lengthSkip){
        long numOps=0;
        
        
        int shapelets =0;
        //calculate number of shapelets in a single instance.
        for (int length = minShapeletLength; length <= maxShapeletLength; length+=lengthSkip) {
            
            long shapeletsLength = (numAttributes - length + 1) / posSkip;
            shapelets+=shapeletsLength;
            
            //System.out.println(shapeletsLength);
            
            long shapeletsCompared = (numAttributes - length + 1);
            
            //each shapelet gets compared to all other subsequences, and they make l operations per comparison for every series..
            long comparisonPerSeries = shapeletsLength * shapeletsCompared * length * (numInstances-1);
            
            numOps +=comparisonPerSeries; 
        }

        System.out.println("shapelets2 " + (shapelets*numInstances));
        
        //for every series.
        numOps *= numInstances;
        return numOps;
    }
    
    
    public static long calc(int n, int m, int min, int max, int pos, int len)
    {
        long numOps =0;
        
        //-1 from max because we index from 0.
        for(int length = 0; length <= ((max-min)/len); length++){
                        
            int currentLength = (len*length) + min;
            long shapeletsLength = (long) Math.ceil((double)(m - currentLength + 1) / (double) pos); //shapelts found.
            
            long shapeletsCompared = (m - currentLength + 1);
            
            numOps += shapeletsLength*shapeletsCompared*currentLength;
        }

        numOps*= n * (n-1);
        return numOps;
    }    

    
    public static BigInteger calculateOps(int n, int m, int posS, int lenS){
        long mSqd = m*m;
        long lenSqd = lenS*lenS;
        long nSqd = n*n;

        BigInteger temp1 = new BigInteger(Long.toString(mSqd));
        temp1 = temp1.multiply(new BigInteger(Long.toString(m)));
        BigInteger temp2 = new BigInteger(Long.toString(7*mSqd));
        BigInteger temp3 = new BigInteger(Long.toString(m*(lenSqd - (18*lenS) + 27)));
        BigInteger temp4 = new BigInteger(Long.toString(lenS*((5*lenS) - 24) + 27));
        
        BigInteger bg = new BigInteger("0");
        bg = bg.add(temp1);
        bg = bg.add(temp2);
        bg = bg.subtract(temp3);
        bg = bg.add(temp4);
        bg = bg.multiply(new BigInteger(Long.toString((nSqd-n))));
        bg = bg.multiply(new BigInteger(Long.toString((m-3))));
        
        BigDecimal bg1 = new BigDecimal(bg);
        
        bg1 =bg1.divide(new BigDecimal(Long.toString((12 * posS * lenS))), MathContext.DECIMAL32);

        return bg1.toBigInteger();
    }
    
    
    public static double calculateN(int n, int m, long time){
        long opCount = time / nanoToOp; 
        
        BigDecimal numerator = new BigDecimal(Long.toString(12*opCount));
        
        BigInteger temp1 = new BigInteger(Long.toString(m*m));
        temp1 = temp1.multiply(new BigInteger(Long.toString(m)));
        BigInteger temp2 = new BigInteger(Long.toString(7*m*m));
        BigInteger temp3 = new BigInteger(Long.toString(10*m));
        BigInteger temp4 = new BigInteger(Long.toString(8));
        
        temp1 = temp1.add(temp2);
        temp1 = temp1.subtract(temp3);
        temp1 = temp1.add(temp4);
        temp1 = temp1.multiply(new BigInteger(Long.toString(m-3)));
        
        BigDecimal denominator = new BigDecimal(temp1);

        BigDecimal result = utilities.StatisticalUtilities.sqrt(numerator.divide(denominator, MathContext.DECIMAL32), MathContext.DECIMAL32);
        
        //sqrt result.
        result = result.divide(new BigDecimal(n), MathContext.DECIMAL32);
        
        return Math.min(result.doubleValue(), 1.0); //return the proportion of n.
    }
    
    
    
    public static void main(String[] args) throws IOException
    {                 
        String dirPath = "C:\\LocalData\\Dropbox\\TSC Problems (1)\\";
        File dir  = new File(dirPath);
        for(File dataset : dir.listFiles()){
            if(!dataset.isDirectory()) continue;
            
            String f = dataset.getPath()+ File.separator + dataset.getName() + "_TRAIN.arff";
        
            Instances train = ClassifierTools.loadData(f);
            
            long shapelets = calculateNumberOfShapelets(train, 3, train.numAttributes()-1);
            //long ops = calculateOperations(train, 3, train.numAttributes()-1);
            
            System.out.print(dataset.getName() + ",");
            System.out.print(train.numAttributes()-1 + ",");
            System.out.print(train.numInstances() + ",");
            int min = 3;
            int max = train.numAttributes()-1;
            int pos = 1;
            int len = 1;
            
            /*FullShapeletTransform transform = new FullShapeletTransform();
            transform.setSearchFunction(new ShapeletSearch(min,max,len, pos));
            transform.setSubSeqDistance(new SubSeqDistance());
            transform.supressOutput();
            transform.process(train);
            long ops3 = transform.getCount();*/
            
            long ops4 = calc(train.numInstances(), train.numAttributes()-1, min, max,pos,len);
            
            double n = calculateN(train.numInstances(), train.numAttributes()-1, dayNano);

            
            //calculate n for minimum class rep of 25.
            int small_sf = InstanceTools.findSmallestClassAmount(train);           
            double proportion = 1.0;
            if (small_sf>25){
                proportion = (double)25/(double)small_sf;
            }
            
            
            System.out.print(ops4 + ",");
            System.out.print(n + ",");
            System.out.print(proportion + "\n");
        }
    }
    
}
