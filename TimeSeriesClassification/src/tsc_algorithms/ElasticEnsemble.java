/*
 * A new Elastic Ensemble for sharing with others
 */
package tsc_algorithms;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import tsc_algorithms.elastic_ensemble.DTW1NN;
import tsc_algorithms.elastic_ensemble.ED1NN;
import tsc_algorithms.elastic_ensemble.ERP1NN;
import tsc_algorithms.elastic_ensemble.Efficient1NN;
import tsc_algorithms.elastic_ensemble.LCSS1NN;
import tsc_algorithms.elastic_ensemble.MSM1NN;
import tsc_algorithms.elastic_ensemble.TWE1NN;
import tsc_algorithms.elastic_ensemble.WDTW1NN;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.DerivativeFilter;

/**
 *
 * @author sjx07ngu
 */
public class ElasticEnsemble implements Classifier{

    // utility to enable AJBs COTE 
    double[] previousPredictions = null;

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public enum ConstituentClassifiers{ 
        Euclidean_1NN, 
        DTW_R1_1NN, 
        DTW_Rn_1NN, 
        WDTW_1NN, 
        DDTW_R1_1NN, 
        DDTW_Rn_1NN, 
        WDDTW_1NN, 
        LCSS_1NN, 
        MSM_1NN, 
        TWE_1NN, 
        ERP_1NN
    };
    
    public static boolean isDerivative(ConstituentClassifiers classifier){
        return (classifier==ConstituentClassifiers.DDTW_R1_1NN || classifier==ConstituentClassifiers.DDTW_Rn_1NN || classifier==ConstituentClassifiers.WDDTW_1NN);
    }
    
    public static boolean isFixedParam(ConstituentClassifiers classifier){
        return (classifier==ConstituentClassifiers.DDTW_R1_1NN || classifier==ConstituentClassifiers.DTW_R1_1NN || classifier==ConstituentClassifiers.Euclidean_1NN);
    }
    
    
    private final ConstituentClassifiers[] classifiersToUse;
    private String datasetName;
    private int resampleId;
    private String resultsDir;
    private double[] cvAccs;
    
    private boolean buildFromFile = false;
    private boolean writeToFile = false;
    private Instances train;
    private Instances derTrain;
    private Efficient1NN[] classifiers = null;
    
    private boolean usesDer = false;
    private static DerivativeFilter df = new DerivativeFilter();
    
    /**
     * Default constructor; includes all constituent classifiers
     */
    public ElasticEnsemble(){
        this.classifiersToUse = ConstituentClassifiers.values();
    }
    
    /**
     * Constructor allowing specific constituent classifier types to be passed
     * @param classifiersToUse ConstituentClassifiers[] list of classifiers to use as enums
     */
    public ElasticEnsemble(ConstituentClassifiers[] classifiersToUse){
        this.classifiersToUse = classifiersToUse;
    }
    
    /**
     * Constructor that builds an EE from existing training output. By default includes all constituent classifier types.
     * NOTE: this DOES NOT resample data; data must be resampled independently of the classifier. This just ensures the correct naming convention of output files
     * 
     * @param resultsDir path to the top-level of the stored training output 
     * @param datasetName name of the dataset to be loaded
     * @param resampleId  resampleId of the dataset to be loaded
     */
    public ElasticEnsemble(String resultsDir, String datasetName, int resampleId){
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifiersToUse = ConstituentClassifiers.values();
        this.buildFromFile = true;
    }
    
    /**
     * Constructor that builds an EE from existing training output. Includes the classifier types passed in as an array of enums
     * 
     * @param resultsDir path to the top-level of the stored training output 
     * @param datasetName name of the dataset to be loaded
     * @param resampleId  resampleId of the dataset to be loaded
     * @param classifiersToUse the classifiers to load
     */
    public ElasticEnsemble(String resultsDir, String datasetName, int resampleId, ConstituentClassifiers[] classifiersToUse){
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifiersToUse = classifiersToUse;
        this.buildFromFile = true;
    }
    
    /** 
     * Turns on file writing to store training output. NOTE: this doesn't resample the data; data needs to be resampled independently of the classifier. This just ensures the correct naming convention for output files.
     * 
     * @param resultsDir path to the top-level of the training output store (makes dir if it doesn't exist)
     * @param datasetName identifier in the written files for this dataset
     * @param resampleId  resample id of the dataset
     */
    public void setFileWritingOn(String resultsDir, String datasetName, int resampleId){
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.writeToFile = true;
    }
    
    /**
     * Builds classifier. If building from file, cv weights and predictions will be loaded from file. If running from scratch, training cv will be performed for constituents to find best params, cv accs, and cv preds
     * @param train The training data
     * @throws Exception if building from file and results not found, or if there is an issue with the training data
     */
    @Override
    public void buildClassifier(Instances train) throws Exception{
        this.train = train;
        this.derTrain = null;
        usesDer = false;
        
        classifiers = new Efficient1NN[this.classifiersToUse.length];
        cvAccs = new double[classifiers.length];
        
        for(int c = 0; c < classifiers.length; c++){
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            if(isDerivative(this.classifiersToUse[c])){
                usesDer = true;
            }
        }
        
        if(usesDer){
            this.derTrain = df.process(train);
        }
        
        if(buildFromFile){
            File existingTrainOut;
            Scanner scan;
            int paramId;
            double cvAcc;
            for(int c = 0; c < classifiers.length; c++){
                existingTrainOut = new File(this.resultsDir+classifiersToUse[c]+"/Predictions/"+datasetName+"/trainFold"+this.resampleId+".csv");
                if(!existingTrainOut.exists()){
                    throw new Exception("Error: training file doesn't exist for "+existingTrainOut.getAbsolutePath());
                }
                scan = new Scanner(existingTrainOut);
                scan.useDelimiter("\n");
                scan.next();//header
                paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                cvAcc = Double.parseDouble(scan.next().trim().split(",")[0]);
                
                if(isDerivative(classifiersToUse[c])){
                    if(!isFixedParam(classifiersToUse[c])){
                        classifiers[c].setParamsFromParamId(derTrain, paramId);
                    }
                    classifiers[c].buildClassifier(derTrain);
                }else{
                    if(!isFixedParam(classifiersToUse[c])){
                        classifiers[c].setParamsFromParamId(train, paramId);
                    }
                    classifiers[c].buildClassifier(train);
                }
                cvAccs[c] = cvAcc;
            }
        }else{
                
            for(int c = 0; c < classifiers.length; c++){
                if(writeToFile){
                    classifiers[c].setFileWritingOn(this.resultsDir, this.datasetName, this.resampleId);
                }
                if(isDerivative(classifiersToUse[c])){
                    cvAccs[c] = classifiers[c].loocv(derTrain)[0];
                }else{
                    cvAccs[c] = classifiers[c].loocv(train)[0];
                }
            }
        }
    }
    
    /**
     * Returns an Efficient1NN object corresponding to the input enum. Output classifier includes the correct internal information for handling LOOCV/param tuning.
     * @param classifier
     * @return
     * @throws Exception 
     */
    public static Efficient1NN getClassifier(ConstituentClassifiers classifier) throws Exception{
        Efficient1NN knn = null;
        switch(classifier){
            case Euclidean_1NN:
                return new ED1NN();
            case DTW_R1_1NN:
                return new DTW1NN(1);
            case DDTW_R1_1NN:
                knn = new DTW1NN(1);
                knn.setClassifierIdentifier(classifier.toString());
                return knn;
            case DTW_Rn_1NN:
                return new DTW1NN();
            case DDTW_Rn_1NN:
                knn = new DTW1NN();
                knn.setClassifierIdentifier(classifier.toString());;
                return knn;
            case WDTW_1NN:
                return new WDTW1NN();
            case WDDTW_1NN:
                knn = new WDTW1NN();
                knn.setClassifierIdentifier(classifier.toString());
                return knn;
            case LCSS_1NN:
                return new LCSS1NN();
            case ERP_1NN:
                return new ERP1NN();
            case MSM_1NN:
                return new MSM1NN();
            case TWE_1NN:
                return new TWE1NN();
            default: 
                throw new Exception("Unsupported classifier type");
        }
            
    }
    
    /**
     * Classify a test instance. Each constituent classifier makes a prediction, votes are weighted by CV accs, and the majority weighted class value vote is returned
     * @param instance test instance
     * @return predicted class value of instance
     * @throws Exception 
     */
    public double classifyInstance(Instance instance) throws Exception{
        if(classifiers==null){
            throw new Exception("Error: classifier not built");
        }
        Instance derIns = null;
        if(this.usesDer){
            Instances temp = new Instances(derTrain,1);
            temp.add(instance);
            temp = df.process(temp);
            derIns = temp.instance(0);
        }
        
        double bsfVote = -1;
        double[] classTotals = new double[train.numClasses()];
        ArrayList<Double> bsfClassVal = null;
        
        double pred;
        this.previousPredictions = new double[this.classifiers.length];
        
        for(int c = 0; c < classifiers.length; c++){
            if(isDerivative(classifiersToUse[c])){
                pred = classifiers[c].classifyInstance(derIns);
            }else{
                pred = classifiers[c].classifyInstance(instance);
            }
            previousPredictions[c] = pred;
            
            try{
                classTotals[(int)pred] += cvAccs[c];
            }catch(Exception e){
                System.out.println("cv accs "+cvAccs.length);
                System.out.println(pred);
                throw e;
            }
            
            if(classTotals[(int)pred] > bsfVote){
                bsfClassVal = new ArrayList<>();
                bsfClassVal.add(pred);
                bsfVote = classTotals[(int)pred];
            }else if(classTotals[(int)pred] == bsfVote){
                bsfClassVal.add(pred);
            }
        }
        
        if(bsfClassVal.size()>1){
            return bsfClassVal.get(new Random(46).nextInt(bsfClassVal.size()));
        }
        return bsfClassVal.get(0);
    }
    
    public double[] getPreviousPredictions() throws Exception{
        if(this.previousPredictions == null){
            throw new Exception("Error: no previous instance found");
        }
        return this.previousPredictions;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        if(classifiers==null){
            throw new Exception("Error: classifier not built");
        }
        Instance derIns = null;
        if(this.usesDer){
            Instances temp = new Instances(derTrain,1);
            temp.add(instance);
            temp = df.process(temp);
            derIns = temp.instance(0);
        }
        
        double[] classTotals = new double[train.numClasses()];
        double cvSum = 0;
        double pred;
        
        for(int c = 0; c < classifiers.length; c++){
            if(isDerivative(classifiersToUse[c])){
                pred = classifiers[c].classifyInstance(derIns);
            }else{
                pred = classifiers[c].classifyInstance(instance);
            }
            try{
                classTotals[(int)pred] += cvAccs[c];
            }catch(Exception e){
                System.out.println("cv accs "+cvAccs.length);
                System.out.println(pred);
                throw e;
            }
            cvSum+=cvAccs[c];
        }
        
        for(int c = 0; c < classTotals.length; c++){
            classTotals[c]/=cvSum;
        }
        
        return classTotals;
    }
    
    public double[] getCVAccs() throws Exception{
        if(this.cvAccs==null){
            throw new Exception("Error: classifier not built yet");
        }
        return this.cvAccs;
    }
    
    
    private String getClassifierInfo(){
        StringBuilder st = new StringBuilder();
        st.append("EE using:\n");
        st.append("=====================\n");
        for(int c = 0; c < classifiers.length; c++){
            st.append(classifiersToUse[c]).append(" ").append(classifiers[c].getClassifierIdentifier()).append(" ").append(cvAccs[c]).append("\n");
        }
        return st.toString();
    }
    

    @Override
    public String toString(){
        return super.toString()+"\n"+this.getClassifierInfo();
    }
 
    
    public static void exampleUsage(String datasetName, int resampeId, String outputResultsDirName) throws Exception{
        
        
        
    }
    
    public static void main(String[] args) throws Exception{

        
    }
}

