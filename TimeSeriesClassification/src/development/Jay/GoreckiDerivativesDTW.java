package development.Jay;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.DTW;
import weka.core.elastic_distance_measures.DTW_DistanceBasic;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class GoreckiDerivativesDTW extends GoreckiDerivativesEuclideanDistance{
    
    public static final String DATA_DIR = "C:/Temp/Dropbox/TSC Problems/";
    
    public static final double[] ALPHAS = {
        //<editor-fold defaultstate="collapsed" desc="alpha values">
        1,
        1.01,
        1.02,
        1.03,
        1.04,
        1.05,
        1.06,
        1.07,
        1.08,
        1.09,
        1.1,
        1.11,
        1.12,
        1.13,
        1.14,
        1.15,
        1.16,
        1.17,
        1.18,
        1.19,
        1.2,
        1.21,
        1.22,
        1.23,
        1.24,
        1.25,
        1.26,
        1.27,
        1.28,
        1.29,
        1.3,
        1.31,
        1.32,
        1.33,
        1.34,
        1.35,
        1.36,
        1.37,
        1.38,
        1.39,
        1.4,
        1.41,
        1.42,
        1.43,
        1.44,
        1.45,
        1.46,
        1.47,
        1.48,
        1.49,
        1.5,
        1.51,
        1.52,
        1.53,
        1.54,
        1.55,
        1.56,
        1.57
//</editor-fold>
    };
    public static final String[] GORECKI_DATASETS = {
        //<editor-fold defaultstate="collapsed" desc="Datasets from the paper">
        "fiftywords", // 450,455,270,50
        "Adiac", // 390,391,176,37
        "Beef", // 30,30,470,5
        "CBF", // 30,900,128,3
        "Coffee", // 28,28,286,2
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "fish", // 175,175,463,7
        "GunPoint", // 50,150,150,2
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
        "SwedishLeaf", // 500,625,128,15
        "SyntheticControl", // 300,300,60,6
        "Trace", // 100,100,275,4
        "TwoPatterns", // 1000,4000,128,4
        "wafer", // 1000,6164,152,2
        "yoga"// 300,3000,426,2
        //</editor-fold>
    };
        
//    private double alpha;
//    private double a;
//    private double b;

    
    public GoreckiDerivativesDTW(){
        super();
    }
    public GoreckiDerivativesDTW(Instances train){
        super(train);
    }
    public GoreckiDerivativesDTW(double alpha){
        super(alpha);
    }
    
    public GoreckiDerivativesDTW(double a, double b){
        super(a,b);
    }

    @Override
    public double distance(Instance one, Instance two){
        return this.distance(one, two, Double.MAX_VALUE);
    }

    @Override
    public double distance(Instance one, Instance two, double cutoff, PerformanceStats stats){
        return this.distance(one,two,cutoff);
    }

    @Override
    public double distance(Instance first, Instance second, double cutoff){
        
        double[] distances = getNonScaledDistances(first, second);
        return a*distances[0]+b*distances[1];
    }

    public double[] getNonScaledDistances(Instance first, Instance second){
        double dist = 0;
        double derDist = 0;

//        DTW dtw = new DTW();
        DTW_DistanceBasic dtw = new DTW_DistanceBasic();
        int classPenalty = 0;
        if(first.classIndex()>0){
            classPenalty=1;
        }
        
        GoreckiDerivativeFilter filter = new GoreckiDerivativeFilter();
        Instances temp = new Instances(first.dataset(),0);
        temp.add(first);
        temp.add(second);
        try{
            temp = filter.process(temp);
        }catch(Exception e){
            e.printStackTrace();
            return null;
        }        
        
        dist = dtw.distance(first, second);
        derDist = dtw.distance(temp.get(0), temp.get(1), Double.MAX_VALUE);
        
        return new double[]{Math.sqrt(dist),Math.sqrt(derDist)};
    }

    

    public static void recreateGoreckiTable() throws Exception{
        recreateGoreckiTable(0);
    }
    
    
    public static void recreateGoreckiTable(int seed) throws Exception{
        
        String[] datasets = GORECKI_DATASETS;
//        String[] datasets = {"GunPoint"};
        
        String clusterDataDir = "../TSC Problems/";
//        String clusterDataDir = DATA_DIR;
        
        Instances train, test, dTrain, dTest;
        EuclideanDistance ed;
        kNN knn;
        int correct;
        double acc, err;
        DecimalFormat df = new DecimalFormat("##.##");
        
        // important - use the correct one! Gorecki uses different derivatives to Keogh
        GoreckiDerivativeFilter derFilter = new GoreckiDerivativeFilter();
        
        
        StringBuilder st = new StringBuilder();
        st.append("Dataset,ED,DED,DD_ED,DTW,DDTW,DD_DTW\n");
        
        
        // quick proof of concept to make sure we can access all of the ARFFS (mine should be fine, Tony's may have lowercase names that break it if they're not changed)
        // add a bit of time on to the complete run time, but better than breaking at the end on wafer and not printing anything!
        for(String dataset:datasets){
            Instances temp = ClassifierTools.loadData(clusterDataDir+dataset+"/"+dataset+"_TRAIN");
            temp = ClassifierTools.loadData(clusterDataDir+dataset+"/"+dataset+"_TEST");
        }
        
        // also, while we're at it check that we can write to the desired output file
        new File("goreckiResampleResults").mkdir();
        FileWriter outputFile = new FileWriter("goreckiResampleResults/seed_"+seed+".txt");
        outputFile.close();
        
        for(String dataset:datasets){
            
            st.append(dataset+",");
            
            train = ClassifierTools.loadData(clusterDataDir+dataset+"/"+dataset+"_TRAIN");
            test = ClassifierTools.loadData(clusterDataDir+dataset+"/"+dataset+"_TEST");
            
            // instance resampling happens here, seed of 0 means that the standard train/test split is used
            if(seed!=0){
                Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, seed);
                train = temp[0];
                test = temp[1];
            }
            
            dTrain = derFilter.process(train);
            dTest = derFilter.process(test);
            
            // ED 
            ed = new GoreckiEuclideanDistance();
            ed.setDontNormalize(true);
            knn = new kNN(ed);
            correct = getCorrect(knn, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            st.append(df.format(err)+",");
            
            // DED
            ed = new GoreckiEuclideanDistance();
            knn = new kNN(ed);
            correct = getCorrect(knn, dTrain, dTest);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            st.append(df.format(err)+",");
            
            // their alpha stuff is incorrect, use scaled a and b between 0:0.01:1 and 1:-0.01:0
//          // DDED
            GoreckiDerivativesEuclideanDistance gdAB = new GoreckiDerivativesEuclideanDistance(train); 
            knn = new kNN(gdAB);
            correct = getCorrect(knn, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            st.append(df.format(err)+",");
            
            //DTW
            DTW_DistanceBasic dtw = new DTW_DistanceBasic();
            knn = new kNN(dtw);
            correct = getCorrect(knn, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            st.append(df.format(err)+",");
            
            // DDTW
            DTW_DistanceBasic dDtw = new DTW_DistanceBasic();
            knn = new kNN(dDtw);
            correct = getCorrect(knn, dTrain, dTest);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            st.append(df.format(err)+",");
            
            // DDDTW
            GoreckiDerivativesDTW gdtw = new GoreckiDerivativesDTW(train);
            knn = new kNN(gdtw);
            correct = getCorrect(knn, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            st.append(df.format(err)+"\n");
            
        }
        
        outputFile = new FileWriter("goreckiResampleResults/seed_"+seed+".txt");
        outputFile.append(st);
        outputFile.close();
        
        
    }
    
    private enum GoreckiClassifierType{ED,DED,DD_ED,DTW,DDTW,DD_DTW};
    
    
    
    // They calculate derivatives differently to the transform we have (which matches Keogh et al.'s DDTW implemetation)
    // Derivatives are built into the new distance measures, but this is needed to recreating the derivative Euclidean/DTW comparison results 
    private static class GoreckiDerivativeFilter extends weka.filters.SimpleBatchFilter{

        @Override
        public String globalInfo() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
            
            Instances output = new Instances(inputFormat,0);
            output.deleteAttributeAt(0);
            output.setRelationName("goreckiDerivative_"+output.relationName());
            for(int a = 0; a < output.numAttributes()-1; a++){
                output.renameAttribute(a, "derivative_"+a);
            }
            
            return output;
            
        }

        @Override
        public Instances process(Instances instances) throws Exception {
            
            Instances output = determineOutputFormat(instances);
            Instance thisInstance;
            Instance toAdd;
            double der;
            for(int i = 0; i < instances.numInstances(); i++){
                thisInstance = instances.get(i);
                toAdd = new DenseInstance(output.numAttributes());
                for(int a = 1; a < instances.numAttributes()-1; a++){
                    der = thisInstance.value(a)-thisInstance.value(a-1);
                    toAdd.setValue(a-1, der);
                }
                toAdd.setValue(output.numAttributes()-1, thisInstance.classValue());
                output.add(toAdd);
            }
            return output;
        }
        
    }
    
    public static void scriptmaker_goreckiResamples(){
        // make dirs
        new File("goreckiResampleScripts").mkdir();
        
        
        
    }
    
    public static void main(String[] args) throws Exception{
        recreateGoreckiTable();
    }
    
}
