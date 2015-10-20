package development;

/***********************************************************************************************************************************
HOW TO USE

Priors:
    -   ouputIdentifier: Select an identifier for the experiments to be carried out. A sensible option is to use the dataset name,
        e.g. ItalyPowerDemand, Beef, etc. This is not limited to the dataset name however to facilitate running different data
        partitions, different training folds, resampling experiments, etc. The only specific requirements are that the identifier
        must be a valid String with no white-space characters, and it must be consistent between method calls to ensure the correct
        are used to build the classifier.

1)  CLUSTER: Generate individual results for all classifiers. Method: writeCvResultsForCluster (wrapped in simulateClusterRun as an example)
2)  CLUSTER: After 1 is fully completed, parse results to select best parameter options and condense results into a single file for each
             classifier (optional: method can remove individual results once parsed). See method: writeAllBestParamFiles
3)  LOCAL:   Build classifier using the parsed output. The classifier must be instantiated with the outputIdentifier as a parameter,
             and the parsed output must be in <projectDir>/eeClusterOutput/<outputIdentifier>_parsedResults/.

See method exampleUseCase() for a local example of all work processing. The only difference when running on GRACE is that the correct
BSUB files must be created and submitted. Instructions on this initial step to follow.


***********************************************************************************************************************************/

import development.Jay.ElasticEnsembleBakeOffParser;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;
import java.util.TreeMap;

import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;

import tsc_algorithms.ElasticEnsemble;
import weka.classifiers.lazy.kNN;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.EuclideanDistance;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.ERPDistance;
import weka.core.elastic_distance_measures.LCSSDistance;
import weka.core.elastic_distance_measures.MSMDistance;
import weka.core.elastic_distance_measures.SakoeChibaDTW;
import weka.core.elastic_distance_measures.TWEDistance;
import weka.core.elastic_distance_measures.WeightedDTW;
import weka.filters.timeseries.DerivativeFilter;

/**
 *
 * @author sjx07ngu
 */
public class ElasticEnsembleCluster extends ElasticEnsemble{

    private final int resampleId;
    private kNN[] builtClassifiers;
    private String pathToTrainingResults;

    private static String[] datasets = development.DataSets.fileNames;
    


    public ElasticEnsembleCluster(String datasetName, int resampleId){
        super();
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.builtClassifiers = null;
        this.pathToTrainingResults = "";
    }

    public void setPathToTrainingResults(String path){
        this.pathToTrainingResults = path;
    }
    
    private static boolean isParameterised(ClassifierVariants c){
        return !(c==ClassifierVariants.Euclidean_1NN || c == ClassifierVariants.DTW_R1_1NN || c== ClassifierVariants.DDTW_R1_1NN);
    }

    private void loadCvFromFile(Instances train) throws Exception{

        // 1. check that results are available for all classifiers
        String[] line;

        this.builtClassifiers = new kNN[finalClassifierTypes.length];
        
        for(int c = 0; c < this.finalClassifierTypes.length; c++){
            // should be in the form of:
            //      line 1: cv accuracy
            //      line 2: bestParamId - utility method included to convert this into the correct representation for a given classifier
            //      line 3 to m+2: prediction/actualClassValue (second part is redundant, but used as a validation measure)

//            File resultFile = new File("eeClusterOutput/"+this.inputNameIdentifier+"/"+this.inputNameIdentifier+"_parsedOutput/"+this.inputNameIdentifier+"_"+finalClassifiers[c]+".txt");
            File resultFile = new File(pathToTrainingResults+"eeClusterOutput/"+this.datasetName+"/"+this.datasetName+"_"+this.resampleId+"/"+this.datasetName+"_"+this.resampleId+"_parsedOutput/"+this.datasetName+"_"+this.resampleId+"_"+finalClassifierTypes[c]+".txt");
            if(!resultFile.exists()){
//                throw new Exception("Error: result file could not be loaded - "+"eeClusterOutput/"+this.inputNameIdentifier+"/"+this.inputNameIdentifier+"_parsedOutput/"+this.inputNameIdentifier+"_"+finalClassifiers[c]+".txt");
                throw new Exception("Error: result file could not be loaded - "+pathToTrainingResults+"eeClusterOutput/"+this.datasetName+"/"+this.datasetName+"_"+this.resampleId+"/"+this.datasetName+"_"+this.resampleId+"_parsedOutput/"+this.datasetName+"_"+this.resampleId+"_"+finalClassifierTypes[c]+".txt");
            }
            Scanner scan = new Scanner(resultFile);
            scan.useDelimiter("\n");
            this.cvAccs[c] = Double.parseDouble(scan.next().trim())*100; // original implementation is 0-100, not 0-1, so need to correct to match formatting
            this.bestParams[c] = getParamsFromParamId(this.finalClassifierTypes[c], Integer.parseInt(scan.next().trim()), train);
//            System.out.println(this.bestParams[c][0]);
            for(int i = 0; i < train.numInstances(); i++){
                line = scan.next().split("/");
                this.cvPreds[c][i] = Double.parseDouble(line[0].trim());
                if(Double.parseDouble(line[1].trim())!=train.instance(i).classValue()){
                    throw new Exception("Error: class mismatch issue for instance "+(i+1)+". File: "+line[1]+", Instances: "+train.instance(i).classValue());
                }
            }
            this.builtClassifiers[c] = getInternalClassifier(this.finalClassifierTypes[c], this.bestParams[c], train);
        }

    }


    @Override
    public void buildClassifier(Instances train) throws Exception {

//        if(this.inputNameIdentifier == null){
//            throw new Exception("Error: this classifier must be built using pre-exitsing crossvalidation results. Please set input identifier and ensure files are in the eeClusterResults dir of this project's files.");
//        }
        if(this.datasetName == null || this.resampleId < 0){
            throw new Exception("Error: this classifier must be built using pre-exitsing crossvalidation results. Please set input identifier and ensure files are in the eeClusterResults dir of this project's files.");
        }

        this.finalClassifierTypes = new ClassifierVariants[classifiersToUse.size()];
        this.classifiersToUse.toArray(finalClassifierTypes);
        this.cvAccs = new double[this.finalClassifierTypes.length];
        this.cvPreds = new double[this.finalClassifierTypes.length][train.numInstances()];
        this.bestParams = new double[this.finalClassifierTypes.length][];
        this.fullTrainingData = train;
        // main work is done here - load in the best cv accuracies, params etc from file
        this.loadCvFromFile(train);

        
        
        
        if(this.ensembleType==EnsembleType.Signif){
            mcNemarsInclusion = this.getMcNemarsInclusion();
        }
        this.classifierBuilt = true;
    }

    /**
     * A method to return a classifier with the desired parameters, derived from a single int for the paramId. This is based on the method
     * with the same name from ElasticEnsemble.java that passes double[] for params instead.
     *
     * @param classifierType ClassifierVarient. The type of elastic distance measure to use with the nearest neighbour.
     * @param paramId int. Used to derive the parameters for the classifier via a private utility method
     * @param trainingData Instances. The data used to train the classifier
     * @return kNN. The nearest neighbour classifier built with the correct parameter and measure settings
     * @throws Exception
     */
    public static kNN getInternalClassifier(ClassifierVariants classifierType, int paramId, Instances trainingData) throws Exception{

        EuclideanDistance distanceMeasure = null;
        kNN knn;
        switch(classifierType){
            case Euclidean_1NN:
                distanceMeasure = new EuclideanDistance();
                distanceMeasure.setDontNormalize(true);
                break;
            case DTW_R1_1NN:
            case DDTW_R1_1NN:
                distanceMeasure = new BasicDTW();
                break;
            case DTW_Rn_1NN:
            case DDTW_Rn_1NN:
                distanceMeasure = new SakoeChibaDTW(((double)paramId)/100);
                break;
            case WDTW_1NN:
            case WDDTW_1NN:
                distanceMeasure = new WeightedDTW(((double)paramId)/100);
                break;
            case LCSS_1NN:
                double stdTrain = LCSSDistance.stdv_p(trainingData);
                double stdFloor = stdTrain*0.2;
                double[] epsilons = LCSSDistance.getInclusive10(stdFloor, stdTrain);
                int[] deltas = LCSSDistance.getInclusive10(0, (trainingData.numAttributes()-1)/4);

                distanceMeasure = new LCSSDistance(deltas[paramId/10], epsilons[paramId%10]);
                break;
            case MSM_1NN:
                distanceMeasure = new MSMDistance(msmParms[paramId]);
                break;
            case TWE_1NN:
                distanceMeasure = new TWEDistance(twe_nuParams[paramId/10],twe_lamdaParams[paramId%10]);
                break;
            case ERP_1NN:
                double[] windowSizes = ERPDistance.getInclusive10(0, 0.25);
                double stdv = ERPDistance.stdv_p(trainingData);
                double[] gValues = ERPDistance.getInclusive10(0.2*stdv, stdv);

                distanceMeasure = new ERPDistance(gValues[paramId/10], windowSizes[paramId%10]);
                break;
            default:
                throw new Exception("Error: "+classifierType+" is not a supported classifier type. Please update code to use this in the ensemble");
        }

        knn = new kNN();
        knn.setDistanceFunction(distanceMeasure);
        knn.buildClassifier(trainingData);
        return knn;
    }

    // overridden to make sure code is running in a single core. Also, changed how classifiers are stored in the ensemble, as it currently makes a 
    // new instance of a classifier for each test prediction (for some reason?!)
    @Override
    public double classifyInstance(Instance test) throws Exception{
        
        double[] predictions = new double[this.builtClassifiers.length];
        for(int p=0; p < predictions.length; p++){
            predictions[p] = this.builtClassifiers[p].classifyInstance(test);
        }
        
        switch(this.ensembleType){
            case Best:
                return this.classifyInstances_best(predictions);
            case Equal:
                return this.classifyInstances_equal(predictions);
            case Prop:
            case Signif:
                return this.classifyInstances_prop(predictions);
            default:
                throw new Exception("Error: Unexpected ensemble type");
        }
    }
  
    
    
    public kNN getBuiltClassifier(ClassifierVariants classifierType) throws Exception{
        if(!this.classifierBuilt){
            throw new Exception("Error: this ensemble has not been built yet");
        }
        for(int c = 0; c < this.finalClassifierTypes.length; c++){
            if(this.finalClassifierTypes[c]==classifierType){
                return this.builtClassifiers[c];
            }
        }
        throw new Exception("Error: "+classifierType+" hasn't been included in this ensemble");
    }
    
    
    public static double createEETrainTestOutputFromFiles(String tscProbsDir, String datasetName, int resampleId, String individualClassifierTrainingResultsDir, String individualClassifierTestResultsDir) throws Exception{
 
        ElasticEnsembleCluster ee = new ElasticEnsembleCluster(datasetName, resampleId);
        ee.turnAllClassifiersOn();
        ee.setPathToTrainingResults(individualClassifierTrainingResultsDir);
        
//        String newArffName = ElasticEnsembleBakeOffParser.getNewName(datasetName);
        String newArffName = getNewName(datasetName);

        Instances origTrain = ClassifierTools.loadData(tscProbsDir+newArffName+"/"+newArffName+"_TRAIN");
        Instances origTest = ClassifierTools.loadData(tscProbsDir+newArffName+"/"+newArffName+"_TEST");
        Instances[] resampled = InstanceTools.resampleTrainAndTestInstances(origTrain, origTest, resampleId);
        
        Instances train = resampled[0];
        Instances test = resampled[1];
        
        ee.buildClassifier(train);
        
        
        // test classification
        double[][] testPreds = new double[ee.finalClassifierTypes.length][test.numInstances()];
        
        String classifierName;
        File resultFile;
        Scanner scan;
        int insId;
        double pred, actual;
        String[] lineParts;
        for(int c = 0; c < ee.finalClassifierTypes.length; c++){
            classifierName = ee.finalClassifierTypes[c].toString();
            resultFile = new File(individualClassifierTestResultsDir+"eeClusterOutput_testResults/"+datasetName+"/"+datasetName+"_"+resampleId+"/"+datasetName+"_"+resampleId+"_"+classifierName+".txt");
            scan = new Scanner(resultFile);
            scan.useDelimiter("\n");
            scan.next();// first line is the test acc for the individual classifier, skip
            insId = 0;
            while(scan.hasNext()){
                lineParts=scan.next().split("\\[")[0].split("/");
                pred = Double.parseDouble(lineParts[0].trim());
                actual = Double.parseDouble(lineParts[1].trim());
//                System.out.println(pred+" -- "+actual+"--"+test.instance(insId).classValue());
                if(actual!=test.instance(insId).classValue()){
                    throw new Exception("Error: class val mismatch for "+datasetName+"_"+resampleId+"_"+classifierName+" on instance "+(insId+1));
                }
                testPreds[c][insId] = pred;
                insId++;
            }
            scan.close();
        }
        
        // all test predictions stored in [classifierId][insId], so now we can call classifyInstance on each one
        // Note: only doing prop here, as that's all we're using going forward
        
        int correct = 0;
        double[] thisInsPreds;
        
        for(int i = 0; i < test.numInstances(); i++){
            
            thisInsPreds = new double[testPreds.length];
            for(int c = 0; c < testPreds.length; c++){
                thisInsPreds[c] = testPreds[c][i];
            }
            pred = ee.classifyInstances_prop(thisInsPreds);
            if(pred==test.instance(i).classValue()){
                correct++;
            }
        }
        return (double)correct/test.numInstances();
        
        
        
        
        
    }
    
    
    /**
     * Main method
     *
     * @param args String[] Arguments used to invoke the main logic of the program.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception{
//        hamInvestigation();
        eeClassification();
    
    }
    
//    public static void hamInvestigation() throws Exception{
//        
////        development.Jay.ThreadedExperimentEE.writeCvAllParams(null, null, resampleId, ClassifierVariants.DTW_R1_1NN, null);
//        
//        Instances hamTrain = ClassifierTools.loadData("C:/Temp/Dropbox/TSC Problems/Ham/Ham_TRAIN");
//        Instances hamTest = ClassifierTools.loadData("C:/Temp/Dropbox/TSC Problems/Ham/Ham_TEST");
//        InstanceTools.resampleTrainAndTestInstances(hamTrain, hamTrain, 1);
//        
//    }
    
    public static String getNewName(String oldName){
        
        String newName = oldName.replaceAll("_", "");
        
        if(newName.equalsIgnoreCase("yoga")){
            newName = "Yoga";
        }else if(newName.equalsIgnoreCase("fiftywords")){
            newName = "FiftyWords";
        }else if(newName.equalsIgnoreCase("fish")){
            newName = "Fish";
        }else if(newName.equalsIgnoreCase("MALLAT")){
            newName = "Mallat";
        }else if(newName.equalsIgnoreCase("wafer")){
            newName = "Wafer";
        }else if(newName.equalsIgnoreCase("SonyAIBORobotSurface")){
            newName = "SonyAIBORobotSurface1";
        }else if(newName.equalsIgnoreCase("SonyAIBORobotSurfaceII")){
            newName = "SonyAIBORobotSurface2";
        }
        
        return newName;
    }
    
    public static void eeClassification() throws Exception{
//        development.Jay.ElasticEnsembleClusterExperiments.clusterMaster(args);
        
//        String datasetName = "GunPoint";
        
        String trainingResultsDir = "C:/Jay/2015/FinalEENormalisationCheck/Merged/";
        String testResultsDir = "C:/Jay/2015/FinalEENormalisationCheck_test/";
        
        File[] datasetDirs = new File("C:/Jay/2015/FinalEENormalisationCheck_test/eeClusterOutput_testResults").listFiles();
//        for(File dataDir:datasetDirs){
        StringBuilder missingResults = new StringBuilder();
//        for(int d = datasetDirs.length-8; d < datasetDirs.length; d++){
//        for(int d = datasetDirs.length-8; d < datasetDirs.length-6; d++){
        for(int d = 0; d < datasetDirs.length; d++){
            
//            String datasetName = dataDir.getName();
            String datasetName = datasetDirs[d].getName();
            
//            if(datasetName.equalsIgnoreCase("UWaveGestureLibrary_Y")||datasetName.equalsIgnoreCase("UWaveGestureLibrary_Z")){
//                continue;
//            }
            if(!datasetName.equalsIgnoreCase("Fish")){
//            if(!datasetName.equalsIgnoreCase("ItalyPowerDemand")){
//            if(datasetName.equalsIgnoreCase("Ham") || datasetName.equalsIgnoreCase("Herring") || datasetName.charAt(0)<'U'){
//            if(!datasetName.equalsIgnoreCase("Strawberry")){
                continue;
            }
            System.out.print(getNewName(datasetName)+",");
//            for(int i = 1; i <= 100; i++){
//            for(int i = 65; i <= 65; i++){
            for(int i = 0; i <= 100; i++){
                try{
//                    System.out.print(createEETrainTestOutputFromFiles("C:/Temp/Dropbox/TSC Problems/", datasetName, i, trainingResultsDir, testResultsDir)+",");
                    System.out.print(createEETrainTestOutputFromFiles("C:/Temp/Dropbox/TSC Problems/", datasetName, 65, trainingResultsDir, testResultsDir)+",");
                }catch(Exception e){
                    missingResults.append(e+"\n");
                }
            }
            System.out.println();
        }   
            System.out.println(missingResults);
        
        
        
        System.out.println();
//        double acc = createEETrainTestOutputFromFiles("C:/Temp/Dropbox/TSC Problems/", "ItalyPowerDemand", 1, trainingResultsDir, testResultsDir);
//        System.out.println(acc);
    }
}

//<editor-fold defaultstate="collapsed" desc="Temporary backup while moving cluster logic to a new class">
//package development;
//
///***********************************************************************************************************************************
//HOW TO USE
//
//Priors:
//    -   ouputIdentifier: Select an identifier for the experiments to be carried out. A sensible option is to use the dataset name,
//        e.g. ItalyPowerDemand, Beef, etc. This is not limited to the dataset name however to facilitate running different data
//        partitions, different training folds, resampling experiments, etc. The only specific requirements are that the identifier
//        must be a valid String with no white-space characters, and it must be consistent between method calls to ensure the correct
//        are used to build the classifier.
//
//1)  CLUSTER: Generate individual results for all classifiers. Method: writeCvResultsForCluster (wrapped in simulateClusterRun as an example)
//2)  CLUSTER: After 1 is fully completed, parse results to select best parameter options and condense results into a single file for each
//             classifier (optional: method can remove individual results once parsed). See method: writeAllBestParamFiles
//3)  LOCAL:   Build classifier using the parsed output. The classifier must be instantiated with the outputIdentifier as a parameter,
//             and the parsed output must be in <projectDir>/eeClusterOutput/<outputIdentifier>_parsedResults/.
//
//See method exampleUseCase() for a local example of all work processing. The only difference when running on GRACE is that the correct
//BSUB files must be created and submitted. Instructions on this initial step to follow.
//
//
//***********************************************************************************************************************************/
//
//import java.io.File;
//import java.io.FileWriter;
//import java.text.DecimalFormat;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.Random;
//import java.util.Scanner;
//import java.util.TreeMap;
//
//import utilities.ClassifierTools;
//import utilities.InstanceTools;
//import weka.classifiers.Classifier;
//
//import weka.classifiers.meta.timeseriesensembles.ElasticEnsemble;
//import weka.classifiers.lazy.kNN;
//import weka.core.DistanceFunction;
//import weka.core.Instance;
//import weka.core.Instances;
//import weka.core.EuclideanDistance;
//import weka.core.elastic_distance_measures.BasicDTW;
//import weka.core.elastic_distance_measures.ERPDistance;
//import weka.core.elastic_distance_measures.LCSSDistance;
//import weka.core.elastic_distance_measures.MSMDistance;
//import weka.core.elastic_distance_measures.SakoeChibaDTW;
//import weka.core.elastic_distance_measures.TWEDistance;
//import weka.core.elastic_distance_measures.WeightedDTW;
//import weka.filters.timeseries.DerivativeFilter;
//
///**
// *
// * @author sjx07ngu
// */
//public class ElasticEnsembleCluster extends ElasticEnsemble{
//
//    private final int resampleId;
//    private kNN[] builtClassifiers;
//    private String pathToTrainingResults;
//
//    private static String[] datasets = development.DataSets.fileNames;
//
//
//
//    public ElasticEnsembleCluster(String datasetName, int resampleId){
//        super();
//        this.datasetName = datasetName;
//        this.resampleId = resampleId;
//        this.builtClassifiers = null;
//        this.pathToTrainingResults = "";
//    }
//
//    public void setPathToTrainingResults(String path){
//        this.pathToTrainingResults = path;
//    }
////    public ElasticEnsembleCluster(String datasetName){
////        super();
////        this.datasetName = datasetName;
////        this.resampleId = 0;
////        this.builtClassifiers = null;
////    }
//
//    /**
//     * An example use case of the cluster version for the Elastic Ensemble, run entirely on the local machine
//     *
//     * @param tscDirPath The String path where the main TSC data repository is stored, e.g. "C:/Dropbox/TSC Problems"
//     * @throws Exception
//     */
////    private static void exampleUseCase(String tscDirPath) throws Exception{
////
////        Instances italyTrain = ClassifierTools.loadData(tscDirPath+"ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff");
////        String resultsIdentifier = "italyTrainingExample";
////
////        // 1) Generate individual prediction files for all classifiers with all possible parameter options
////        simulateClusterRun(italyTrain, resultsIdentifier);
////
////        // 2) Parse individual outputs to find the best parameters for each classifier. Boolean option is whether to
////        //    remove individual files once the best are written to file. If set to false, files will not be removed.
////        //    (Note: if the individual files are not removed at this stage, tidyUpIfParsedExists(String identifier)
////        //    can be used to remove individual files later, but only if the parsed output for the classifier exists)
////        writeAllBestParamFiles(resultsIdentifier,true);
////
////
////
////        // 3) Use results to build a classifier
////        ElasticEnsembleCluster eec = new ElasticEnsembleCluster(resultsIdentifier);
////        eec.buildClassifier(italyTrain);
////
////        // 4) Classifier is  equivilent to a locally-built ElasticEnsemble instance, use accordingly.
////        Instances italyTest = ClassifierTools.loadData(tscDirPath+"ItalyPowerDemand/ItalyPowerDemand_TEST.arff");
////        int correct = 0;
////        for(int i = 0; i < italyTest.numInstances(); i++){
////            if(eec.classifyInstance(italyTest.get(i))==italyTest.get(i).classValue()){
////                correct++;
////            }
////        }
////        System.out.println("Accuracy: " + new DecimalFormat("#.###").format((double)correct/italyTest.numInstances()*100)+"%");
////
////    }
//
//    /**
//     * A method to simulate the experiments that would be run on Grace. Creates a dir in the project dir called eeClusterOutput
//     * and generates a subdirectory for each measure type, each parameter option as a subdir of that, and then an individual txt
//     * file for each instance.
//     *
//     * @param instances Instances. The instances used for the cross-validation experiments
//     * @param outputIdentifier String. A consistent identifier for all files relating to this set of experiments
//     * @throws Exception
//     */
//    public static void simulateClusterRun(Instances instances, String outputIdentifier) throws Exception{
//
//        ClassifierVariants[] classifiers = ClassifierVariants.values();
//
//        int numJobs = 100*instances.numInstances();
//        int instanceId, paramId;
//
//        for(int c = 0; c < classifiers.length; c++){
//            for(int j = 0; j < numJobs; j++){   // j is effectively the job number. Need numParamOptions and numInstances to split
//                paramId = j%100;
//                instanceId = j/100;
//                writeCvResultsForCluster(instances, outputIdentifier, classifiers[c], instanceId, paramId);
//                if(!isParameterised(classifiers[c])){ // to avoid running non-parameterised measures multiple times
//                    j+=99;
//                }
//            }
//        }
//    }
//
//    private static boolean isParameterised(ClassifierVariants c){
//        return !(c==ClassifierVariants.Euclidean_1NN || c == ClassifierVariants.DTW_R1_1NN || c== ClassifierVariants.DDTW_R1_1NN);
//    }
//
//    /**
//     * The method that carries out a leave-one-out-cross-validation experiment for a single instance and writes the output to file.
//     * @param instances Instances. The training instances
//     * @param outputNameIdentifier String. A consistent name identifier for all jobs in this experiment
//     * @param measureType ClassifierVariant. The elastic measure used with the nearest neighbour classifier in this experiment
//     * @param instanceId int. The index of the test instance within the training data
//     * @param paramId int. A reference to the job number for deriving which parameter options to use with this classifier
//     * @throws Exception
//     */
//    public static void writeCvResultsForCluster(Instances instances, String outputNameIdentifier, ClassifierVariants measureType, int instanceId, int paramId) throws Exception{
//        Instances train;
//        if(measureType.equals(ClassifierVariants.DDTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_Rn_1NN)||measureType.equals(ClassifierVariants.WDDTW_1NN)){
//            DerivativeFilter d = new DerivativeFilter();
//            train = d.process(instances);
//        }else{
//            train = new Instances(instances);
//        }
//        Instance test = train.remove(instanceId);
//
//        kNN classifier = getInternalClassifier(measureType, paramId, train);
//        double prediction = classifier.classifyInstance(test);
//
//        FileWriter out = null;
//
//        File outDir = new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/");
//        outDir.mkdirs();
//        out = new FileWriter("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"_ins_"+instanceId+".txt");
//        out.append(prediction+"/"+test.classValue());
//        out.close();
//
//        // create a global information file as a contingency (if it doesn't exist). This can provide information such as number of instances and actual class values
//        // without needing to load raw data, and also help inturpret old files if the origin is unclear by storing the relation name
//        String infoLoc = "eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+".info";
//        if(!new File(infoLoc).exists()){
//            out = new FileWriter(infoLoc);
//            String summary = instances.toSummaryString();
//            Scanner scan = new Scanner(summary);
//            scan.useDelimiter("\n");
//            out.append(scan.next().split(":")[1].trim()+"\n"); // relationName
//            out.append(scan.next().split(":")[1].trim()+"\n"); // numInstances
//            scan.close();
//            for(int i = 0; i < instances.numInstances(); i++){
//                out.append(instances.get(i).classValue()+"\n");
//            }
//            out.close();
//        }
//
//    }
//
//
//
//    public static void writeCvVariableFoldSizeAllParamsResultsForCluster(Instances instances, String datasetName, String outputNameIdentifier, ClassifierVariants measureType, int foldSize) throws Exception{
//
//        // don't bother if parsed output already exists for this dataset, resampleId and classifier
//        //TODO
//        if(new File("eeClusterOutput/"+datasetName+"/"+outputNameIdentifier+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+measureType+".txt").exists()){
//            return;
//        }
//
//        // check for parsed output
//
//        Instances fullTrain;
//        double prediction;
//        FileWriter out;
//        int instanceId;
//        File outDir;
//        String outLoc;
//
//        int correct;
//        double acc;
//
//        // set up the number of param options
//        int maxParamId = 100;
//        if(measureType.equals(ClassifierVariants.Euclidean_1NN)||measureType.equals(ClassifierVariants.DTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_R1_1NN)){
//            maxParamId = 1;
//        }
//
//        // transform the data (if necessary). Do this here so we only have to do it once, then copy for other folds
//        if(measureType.equals(ClassifierVariants.DDTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_Rn_1NN)||measureType.equals(ClassifierVariants.WDDTW_1NN)){
//             DerivativeFilter d = new DerivativeFilter();
//             fullTrain = d.process(instances);
//         }else{
//             fullTrain = new Instances(instances);
//         }
//
//        File paramFile;
//
//        for(int paramId = 0; paramId< maxParamId; paramId++){
//
//            outLoc = "eeClusterOutput/"+datasetName+"/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+".txt";
//            paramFile = new File(outLoc);
//            if(paramFile.exists()){
//                if(paramFile.length() > 0){ // weird bug on cluster made a few files with no content (maybe crashed while writing, or didn't close stream)
//                    continue;
//                }
//            }
//
////            outDir = new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/");
//            outDir = new File("eeClusterOutput/"+datasetName+"/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/");
//            outDir.mkdirs();
//
//            int numFolds = (int)Math.ceil((double)fullTrain.numInstances()/foldSize);
//            StringBuilder st = new StringBuilder();
//            Instances train;
//            Instances test;
//
//            correct = 0;
//
//            for(int fold = 0; fold < numFolds; fold++){
//                train = new Instances(fullTrain);
//                test = new Instances(train, 0);
//                for(int i = 0; i < foldSize; i++){
//                    try{
////                        System.out.println("removing "+(fold*foldSize)+", but is actually "+(fold*foldSize+i)+" according to original indexes");
//                        test.add(train.remove(fold*foldSize));
//                    }catch(Exception e){
//                        System.out.println(e);
//                        if(fold!=numFolds-1 || test.isEmpty()){
//                            throw new Exception("Incorrect fold initialisation");
//                        }
//                    }
//                }
//
//
//                kNN classifier = getInternalClassifier(measureType, paramId, train);
//                for(int i = 0; i < test.numInstances();i++){
//                    prediction = classifier.classifyInstance(test.instance(i));
//
////                    classifier.distributionForInstance(null)
//
////                    instanceId = foldSize*fold+i;
////                    outLoc = "eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"_ins_"+instanceId+".txt";
////                    if(new File(outLoc).exists()){
////                        throw new Exception("Error: File already exists: "+outLoc);
////                    }
////                    out = new FileWriter(outLoc);
////                    out.append(prediction+"/"+test.instance(i).classValue());
////                    out.close();
//                    instanceId = foldSize*fold+i;
////                    outLoc = "eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+".txt";
////                    if(new File(outLoc).exists()){
////                        throw new Exception("Error: File already exists: "+outLoc);
////                    }
////                    out = new FileWriter(outLoc);
////                    out.append(instanceId+","+prediction+"/"+test.instance(i).classValue());
////                    out.close();
//                    if(prediction==test.instance(i).classValue()){
//                        correct++;
//                    }
//                    st.append(instanceId+","+prediction+"/"+test.instance(i).classValue()+"\n");
//                }
//
//
//            }
////            outLoc = "eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+".txt";
//            acc = (double)correct/fullTrain.numInstances();
//            out = new FileWriter(outLoc);
//            out.append(acc+"\n");
//            out.append(st);
//            out.close();
//            // create a global information file as a contingency (if it doesn't exist). This can provide information such as number of instances and actual class values
//            // without needing to load raw data, and also help inturpret old files if the origin is unclear by storing the relation name
//            String infoLoc = "eeClusterOutput/"+datasetName+"/"+outputNameIdentifier+"/"+outputNameIdentifier+".info";
//            if(!new File(infoLoc).exists()){
//                out = new FileWriter(infoLoc);
//                String summary = instances.toSummaryString();
//                Scanner scan = new Scanner(summary);
//                scan.useDelimiter("\n");
//                out.append(scan.next().split(":")[1].trim()+"\n"); // relationName
//                out.append(scan.next().split(":")[1].trim()+"\n"); // numInstances
//                scan.close();
//                for(int i = 0; i < instances.numInstances(); i++){
//                    out.append(instances.get(i).classValue()+"\n");
//                }
//                out.close();
//            }
//        }
//    }
//
//    public static void writeCvVariableFoldSizeAllParamsResultsForCluster(Instances trainIn, Instances testIn, String outputNameIdentifier, ClassifierVariants measureType, int foldSize, int resampleSeed) throws Exception{
//
//        Instances instances = InstanceTools.resampleTrainAndTestInstances(trainIn, testIn, resampleSeed)[0];
//
//        Instances fullTrain;
//        double prediction;
//        FileWriter out;
//        int instanceId;
//        String outLoc;
//        File outDir;
//
//        // set up the number of param options
//        int maxParamId = 100;
//        if(measureType.equals(ClassifierVariants.Euclidean_1NN)||measureType.equals(ClassifierVariants.DTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_R1_1NN)){
//            maxParamId = 1;
//        }
//
//        // transform the data (if necessary). Do this here so we only have to do it once, then copy for other folds
//        if(measureType.equals(ClassifierVariants.DDTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_Rn_1NN)||measureType.equals(ClassifierVariants.WDDTW_1NN)){
//             DerivativeFilter d = new DerivativeFilter();
//             fullTrain = d.process(instances);
//         }else{
//             fullTrain = new Instances(instances);
//         }
//
//        for(int paramId = 0; paramId< maxParamId; paramId++){
//
//            outDir = new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/");
//            outDir.mkdirs();
//
//            int numFolds = (int)Math.ceil((double)fullTrain.numInstances()/foldSize);
//
//            Instances train;
//            Instances test;
//            for(int fold = 0; fold < numFolds; fold++){
//                train = new Instances(fullTrain);
//                test = new Instances(train, 0);
//                for(int i = 0; i < foldSize; i++){
//                    try{
//                        System.out.println("removing "+(fold*foldSize)+", but is actually "+(fold*foldSize+i)+" according to original indexes");
//                        test.add(train.remove(fold*foldSize));
//                    }catch(Exception e){
//                        System.out.println(e);
//                        if(fold!=numFolds-1 || test.isEmpty()){
//                            throw new Exception("Incorrect fold initialisation");
//                        }
//                    }
//                }
//
//                kNN classifier = getInternalClassifier(measureType, paramId, train);
//                for(int i = 0; i < test.numInstances();i++){
//                    prediction = classifier.classifyInstance(test.instance(i));
//
//                    instanceId = foldSize*fold+i;
//                    outLoc = "eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"_ins_"+instanceId+".txt";
//                    if(new File(outLoc).exists()){
//                        throw new Exception("Error: File already exists: "+outLoc);
//                    }
//                    out = new FileWriter(outLoc);
//                    out.append(prediction+"/"+test.instance(i).classValue());
//                    out.close();
//                }
//            }
//
//            // create a global information file as a contingency (if it doesn't exist). This can provide information such as number of instances and actual class values
//            // without needing to load raw data, and also help inturpret old files if the origin is unclear by storing the relation name
//            String infoLoc = "eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+".info";
//            if(!new File(infoLoc).exists()){
//                out = new FileWriter(infoLoc);
//                String summary = instances.toSummaryString();
//                Scanner scan = new Scanner(summary);
//                scan.useDelimiter("\n");
//                out.append(scan.next().split(":")[1].trim()+"\n"); // relationName
//                out.append(scan.next().split(":")[1].trim()+"\n"); // numInstances
//                scan.close();
//                for(int i = 0; i < instances.numInstances(); i++){
//                    out.append(instances.get(i).classValue()+"\n");
//                }
//                out.close();
//            }
//        }
//    }
//
//    /**
//     * Once all individual cross-validation experiments have been completer with the writeCvResultsForCluster method, this method parses the results to output files
//     * that determine the best parameters for each classifier, within a set of results with the specofoed outputNameIdentifier
//     *
//     * @param outputNameIdentifier String. A consistent reference for all individual results that form part of the same experiment.
//     * @param measureType ClassifierVariant. The measure that results are to be summarised for.
//     * @param tidyUp boolean. Determines whether to delete individual result files and directories once the best parameter is stored and written to a new output.
//     * @throws Exception
//     */
//    public static void writeBestParamFile(String outputNameIdentifier, ClassifierVariants measureType, boolean tidyUp) throws Exception{
//        writeBestParamFile(null, outputNameIdentifier, measureType, tidyUp);
//    }
//
//    public static void writeBestParamFile(String dataName, String outputNameIdentifier, ClassifierVariants measureType, boolean tidyUp) throws Exception{
//
//        String dirStart;
//        if(dataName!=null){
//            dirStart = "eeClusterOutput/"+dataName+"/"+outputNameIdentifier;
//        }else{
//            dirStart = "eeClusterOutput/"+outputNameIdentifier;
//        }
//
//        if(new File(dirStart+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+measureType+".txt").exists()){
//            System.err.println("Warning: Parsed output already exists for "+outputNameIdentifier+" and "+measureType+". Exiting method without writing to file.");
//            return;
//        }
//
//        int expectedParams;
//        if(measureType.equals(ClassifierVariants.Euclidean_1NN)||measureType.equals(ClassifierVariants.DTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_R1_1NN)){
//            expectedParams = 1;
//        }else{
//            expectedParams =100;
//        }
//
//        double[] paramPredictions;
//        double[] bsfParamPredictions = null;
//
//        int bsfParamId = -1;
//        int correct;
//        int bsfCorrect = -1;
//
//        Scanner scan;
//        String[] line;
//
//        // get infoFile
//        scan = new Scanner(new File(dirStart+"/"+outputNameIdentifier+".info"));
//        scan.useDelimiter("\n");
//        scan.next(); // relationName
//        int expectedInstances = Integer.parseInt(scan.next().trim());
//
//        double[] classVals = new double[expectedInstances];
//        for(int i = 0; i < expectedInstances; i++){
//            classVals[i] = Double.parseDouble(scan.next().trim());
//        }
//        scan.close();
//
//        for(int p = 0; p < expectedParams; p++){ // check accuracy of each parameter
//            correct = 0;
//            paramPredictions = new double[expectedInstances];
//            for(int i = 0; i < expectedInstances; i++){
//                // hardcode filename to be read in to avoid any index issues
//                File results = new File(dirStart+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+p+"/"+outputNameIdentifier+"_"+measureType+"_p_"+p+"_ins_"+i+".txt");
//                if(!results.exists()){
//                    throw new Exception("Error: Missing results for "+measureType+" and paramId "+p+" on instance "+i);
//                }
//                scan = new Scanner(results);
//                line = scan.next().trim().split("/");
//                scan.close();
//                paramPredictions[i] = Double.parseDouble(line[0]);
//                if(paramPredictions[i]==classVals[i]){
//                    correct++;
//                }
//                if(classVals[i]!=Double.parseDouble(line[1])){
//                    throw new Exception("ERROR: class values have been confused. Instance "+i+ "should be "+classVals[i]+", file states "+Double.parseDouble(line[1]));
//                }
//            }
//            if(correct>bsfCorrect){ // favours smaller params/earlioer options. This may cause different paramater selection to original approach for measures with two params - investigate further
//                bsfCorrect = correct;
//                bsfParamId = p;
//                bsfParamPredictions = paramPredictions;
//            }
//        }
//
//        new File(dirStart+"/"+outputNameIdentifier+"_parsedOutput/").mkdirs();
//        FileWriter out = new FileWriter(dirStart+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+measureType+".txt");
//        out.append((double)bsfCorrect/expectedInstances+"\n");
//        out.append(bsfParamId+"\n");
//        for(int i = 0; i < bsfParamPredictions.length; i++){
//            out.append(bsfParamPredictions[i]+"/"+classVals[i]+"\n");
//        }
//        out.close();
//        if(tidyUp){
//            deleteDir(new File(dirStart+"/"+outputNameIdentifier+"_"+measureType));
//        }
//    }
//
//    public static void writeBestParamFileFromClusterOutput(String dataName, String outputNameIdentifier, ClassifierVariants measureType, boolean tidyUp) throws Exception{
//
//        String dirStart = "eeClusterOutput/"+dataName+"/"+outputNameIdentifier;
//
//
//        if(new File(dirStart+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+measureType+".txt").exists()){
////            System.err.println("Parsed output already exists for "+outputNameIdentifier+" and "+measureType+". Exiting method without writing to file.");
//            return;
//        }
//
//        int expectedParams;
//        if(measureType.equals(ClassifierVariants.Euclidean_1NN)||measureType.equals(ClassifierVariants.DTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_R1_1NN)){
//            expectedParams = 1;
//        }else{
//            expectedParams =100;
//        }
//
//        double[] paramPredictions;
//        double[] bsfParamPredictions = null;
//
//        int bsfParamId = -1;
//        int correct;
//        int bsfCorrect = -1;
//
//        double acc;
//        double bsfAcc = -1;
//
//        Scanner scan;
//        String[] linePart1;
//        String[] linePart2;
//
//        // get infoFile
//        scan = new Scanner(new File(dirStart+"/"+outputNameIdentifier+".info"));
//        scan.useDelimiter("\n");
//        scan.next(); // relationName
//        int expectedInstances = Integer.parseInt(scan.next().trim());
//
//        double[] classVals = new double[expectedInstances];
//        for(int i = 0; i < expectedInstances; i++){
//            classVals[i] = Double.parseDouble(scan.next().trim());
//        }
//        scan.close();
//
//        for(int p = 0; p < expectedParams; p++){ // check accuracy of each parameter
//            correct = 0;
//            paramPredictions = new double[expectedInstances];
//            File results = new File(dirStart+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+p+".txt");
//            if(!results.exists()){
//                throw new Exception("Error: Missing result file: "+dirStart+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+p+".txt");
//            }
//
//            scan = new Scanner(results);
//            scan.useDelimiter("\n");
//            acc = Double.parseDouble(scan.next().trim()); // accuracy line
//
//            if(acc > bsfAcc){
//                bsfAcc = acc;
//                for(int i = 0; i < expectedInstances; i++){
//                    linePart1 = scan.next().trim().split(",");
//                    linePart2 = linePart1[1].split("/");
//
//                    // sanity check - make sure we're looking at the correct instance (possible mismatch if fold resampling isn't correct
//                    if(Integer.parseInt(linePart1[0])!=i){
//                        throw new Exception("Instance indexm mismatch. Expected "+i+", line started with "+linePart1[0]);
//                    }
//
//                    // extract predicted and actual (as a further sanity check) from the line
//                    paramPredictions[i] = Double.parseDouble(linePart2[0]);
//                    if(paramPredictions[i]==classVals[i]){
//                        correct++;
//                    }
//                    if(classVals[i]!=Double.parseDouble(linePart2[1])){
//                        throw new Exception("ERROR: class values have been confused. Instance "+i+ "should be "+classVals[i]+", file states "+Double.parseDouble(linePart2[1]));
//                    }
//                }
//            }
//            scan.close();
//
//            if(correct>bsfCorrect){ // favours smaller params/earlioer options. This may cause different paramater selection to original approach for measures with two params - investigate further
//                bsfCorrect = correct;
//                bsfParamId = p;
//                bsfParamPredictions = paramPredictions;
//            }
//        }
//
//        new File(dirStart+"/"+outputNameIdentifier+"_parsedOutput/").mkdirs();
//        FileWriter out = new FileWriter(dirStart+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+measureType+".txt");
//        out.append((double)bsfCorrect/expectedInstances+"\n");
//        out.append(bsfParamId+"\n");
//        for(int i = 0; i < bsfParamPredictions.length; i++){
//            out.append(bsfParamPredictions[i]+"/"+classVals[i]+"\n");
//        }
//        out.close();
//        if(tidyUp){
//            deleteDir(new File(dirStart+"/"+outputNameIdentifier+"_"+measureType));
//        }
//    }
//
//    private static void deleteDir(File dir){
//        if(dir.isDirectory()){
//            File[] files = dir.listFiles();
//            for(int f = 0; f < files.length; f++){
//                deleteDir(files[f]);
//            }
//        }
//        dir.delete();
//    }
//
//    /**
//     *
//     * A utility method for running the writeBestParamFile for all classifiers that are present in the outputNameIdentifier subdir of results
//     *
//     * @param outputNameIdentifier String. A consistent reference for all individual results that form part of the same experiment.
//     * @param tidyUp boolean. Determines whether to delete individual result files and directories once the best parameter is stored and written to a new output.
//     * @throws Exception
//     */
//    public static void writeAllBestParamFiles(String outputNameIdentifier, boolean tidyUp) throws Exception{
//        File[] rawOutputsByClassifier = new File("eeClusterOutput/"+outputNameIdentifier).listFiles();
//        for(int f = 0; f < rawOutputsByClassifier.length; f++){
//            String fileName = rawOutputsByClassifier[f].getName();
//            if(rawOutputsByClassifier[f].isDirectory() && !fileName.contains("parsedOutput")){
//                writeBestParamFile(outputNameIdentifier, ClassifierVariants.valueOf(fileName.substring(outputNameIdentifier.length()+1)), tidyUp);
//            }
//        }
//    }
//
//    /**
//     * A method to clean up individual output files if they remain after parsing. The work is logically identical to running writeBestParamFile with the tidyUp
//     * option set to true.
//     *
//     * @param outputNameIdentifier String. A consistent reference for all individual results that form part of the same experiment.
//     * @throws Exception
//     */
//    public static void tidyUpIfParsedExists(String outputNameIdentifier) throws Exception{
//
//        ClassifierVariants[] classifiers = ClassifierVariants.values();
//        for(int c = 0; c < classifiers.length; c++){
//            if(new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+classifiers[c]+".txt").exists()){
//                // parsed output exists, so we can delete the relevant raw files
//                deleteDir(new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+classifiers[c]));
//            }
//        }
//    }
//
//    // same as the above method, but accounts for different file structure on the cluster when running with jobIds
//    public static void tidyUpIfParsedExistsCluster(String dataset, ClassifierVariants[] classifiers) throws Exception{
//
////        ClassifierVariants[] classifiers = ClassifierVariants.values();
//        for(int i = 1; i <= 100; i++){
//            for(int c = 0; c < classifiers.length; c++){
//                if(new File("eeClusterOutput/"+dataset+"/"+dataset+"_"+i+"/"+dataset+"_"+i+"_parsedOutput/"+dataset+"_"+i+"_"+classifiers[c]+".txt").exists()){
//                    // parsed output exists, so we can delete the relevant raw files
//                    deleteDir(new File("eeClusterOutput/"+dataset+"/"+dataset+"_"+i+"/"+dataset+"_"+i+"_"+classifiers[c]));
//                }else{
//                    System.out.println("parsed data missing for "+dataset+"_"+i+"_"+classifiers[c]);
//                }
//            }
//        }
//    }
//
//
//    private void loadCvFromFile(Instances train) throws Exception{
//
//        // 1. check that results are available for all classifiers
//        String[] line;
//
//        this.builtClassifiers = new kNN[finalClassifierTypes.length];
//
//        for(int c = 0; c < this.finalClassifierTypes.length; c++){
//            // should be in the form of:
//            //      line 1: cv accuracy
//            //      line 2: bestParamId - utility method included to convert this into the correct representation for a given classifier
//            //      line 3 to m+2: prediction/actualClassValue (second part is redundant, but used as a validation measure)
//
////            File resultFile = new File("eeClusterOutput/"+this.inputNameIdentifier+"/"+this.inputNameIdentifier+"_parsedOutput/"+this.inputNameIdentifier+"_"+finalClassifiers[c]+".txt");
//            File resultFile = new File(pathToTrainingResults+"eeClusterOutput/"+this.datasetName+"/"+this.datasetName+"_"+this.resampleId+"/"+this.datasetName+"_"+this.resampleId+"_parsedOutput/"+this.datasetName+"_"+this.resampleId+"_"+finalClassifierTypes[c]+".txt");
//            if(!resultFile.exists()){
////                throw new Exception("Error: result file could not be loaded - "+"eeClusterOutput/"+this.inputNameIdentifier+"/"+this.inputNameIdentifier+"_parsedOutput/"+this.inputNameIdentifier+"_"+finalClassifiers[c]+".txt");
//                throw new Exception("Error: result file could not be loaded - "+pathToTrainingResults+"eeClusterOutput/"+this.datasetName+"/"+this.datasetName+"_"+this.resampleId+"/"+this.datasetName+"_"+this.resampleId+"_parsedOutput/"+this.datasetName+"_"+this.resampleId+"_"+finalClassifierTypes[c]+".txt");
//            }
//            Scanner scan = new Scanner(resultFile);
//            scan.useDelimiter("\n");
//            this.cvAccs[c] = Double.parseDouble(scan.next().trim())*100; // original implementation is 0-100, not 0-1, so need to correct to match formatting
//            this.bestParams[c] = getParamsFromParamId(this.finalClassifierTypes[c], Integer.parseInt(scan.next().trim()), train);
////            System.out.println(this.bestParams[c][0]);
//            for(int i = 0; i < train.numInstances(); i++){
//                line = scan.next().split("/");
//                this.cvPreds[c][i] = Double.parseDouble(line[0].trim());
//                if(Double.parseDouble(line[1].trim())!=train.instance(i).classValue()){
//                    throw new Exception("Error: class mismatch issue for instance "+(i+1)+". File: "+line[1]+", Instances: "+train.instance(i).classValue());
//                }
//            }
//            this.builtClassifiers[c] = getInternalClassifier(this.finalClassifierTypes[c], this.bestParams[c], train);
//        }
//
//    }
//
//
//    @Override
//    public void buildClassifier(Instances train) throws Exception {
//
////        if(this.inputNameIdentifier == null){
////            throw new Exception("Error: this classifier must be built using pre-exitsing crossvalidation results. Please set input identifier and ensure files are in the eeClusterResults dir of this project's files.");
////        }
//        if(this.datasetName == null || this.resampleId < 0){
//            throw new Exception("Error: this classifier must be built using pre-exitsing crossvalidation results. Please set input identifier and ensure files are in the eeClusterResults dir of this project's files.");
//        }
//
//        this.finalClassifierTypes = new ClassifierVariants[classifiersToUse.size()];
//        this.classifiersToUse.toArray(finalClassifierTypes);
//        this.cvAccs = new double[this.finalClassifierTypes.length];
//        this.cvPreds = new double[this.finalClassifierTypes.length][train.numInstances()];
//        this.bestParams = new double[this.finalClassifierTypes.length][];
//        this.fullTrainingData = train;
//        // main work is done here - load in the best cv accuracies, params etc from file
//        this.loadCvFromFile(train);
//
//
//
//
//        if(this.ensembleType==EnsembleType.Signif){
//            mcNemarsInclusion = this.getMcNemarsInclusion();
//        }
//        this.classifierBuilt = true;
//    }
//
//    /**
//     * A method to return a classifier with the desired parameters, derived from a single int for the paramId. This is based on the method
//     * with the same name from ElasticEnsemble.java that passes double[] for params instead.
//     *
//     * @param classifierType ClassifierVarient. The type of elastic distance measure to use with the nearest neighbour.
//     * @param paramId int. Used to derive the parameters for the classifier via a private utility method
//     * @param trainingData Instances. The data used to train the classifier
//     * @return kNN. The nearest neighbour classifier built with the correct parameter and measure settings
//     * @throws Exception
//     */
//    public static kNN getInternalClassifier(ClassifierVariants classifierType, int paramId, Instances trainingData) throws Exception{
//
//        EuclideanDistance distanceMeasure = null;
//        kNN knn;
//        switch(classifierType){
//            case Euclidean_1NN:
//                distanceMeasure = new EuclideanDistance();
//                distanceMeasure.setDontNormalize(true);
//                break;
//            case DTW_R1_1NN:
//            case DDTW_R1_1NN:
//                distanceMeasure = new BasicDTW();
//                break;
//            case DTW_Rn_1NN:
//            case DDTW_Rn_1NN:
//                distanceMeasure = new SakoeChibaDTW(((double)paramId)/100);
//                break;
//            case WDTW_1NN:
//            case WDDTW_1NN:
//                distanceMeasure = new WeightedDTW(((double)paramId)/100);
//                break;
//            case LCSS_1NN:
//                double stdTrain = LCSSDistance.stdv_p(trainingData);
//                double stdFloor = stdTrain*0.2;
//                double[] epsilons = LCSSDistance.getInclusive10(stdFloor, stdTrain);
//                int[] deltas = LCSSDistance.getInclusive10(0, (trainingData.numAttributes()-1)/4);
//
//                distanceMeasure = new LCSSDistance(deltas[paramId/10], epsilons[paramId%10]);
//                break;
//            case MSM_1NN:
//                distanceMeasure = new MSMDistance(msmParms[paramId]);
//                break;
//            case TWE_1NN:
//                distanceMeasure = new TWEDistance(twe_nuParams[paramId/10],twe_lamdaParams[paramId%10]);
//                break;
//            case ERP_1NN:
//                double[] windowSizes = ERPDistance.getInclusive10(0, 0.25);
//                double stdv = ERPDistance.stdv_p(trainingData);
//                double[] gValues = ERPDistance.getInclusive10(0.2*stdv, stdv);
//
//                distanceMeasure = new ERPDistance(gValues[paramId/10], windowSizes[paramId%10]);
//                break;
//            default:
//                throw new Exception("Error: "+classifierType+" is not a supported classifier type. Please update code to use this in the ensemble");
//        }
//
//        knn = new kNN();
//        knn.setDistanceFunction(distanceMeasure);
//        knn.buildClassifier(trainingData);
//        return knn;
//    }
//
//    // overridden to make sure code is running in a single core. Also, changed how classifiers are stored in the ensemble, as it currently makes a
//    // new instance of a classifier for each test prediction (for some reason?!)
//    @Override
//    public double classifyInstance(Instance test) throws Exception{
//
//        double[] predictions = new double[this.builtClassifiers.length];
//        for(int p=0; p < predictions.length; p++){
//            predictions[p] = this.builtClassifiers[p].classifyInstance(test);
//        }
//
//        switch(this.ensembleType){
//            case Best:
//                return this.classifyInstances_best(predictions);
//            case Equal:
//                return this.classifyInstances_equal(predictions);
//            case Prop:
//            case Signif:
//                return this.classifyInstances_prop(predictions);
//            default:
//                throw new Exception("Error: Unexpected ensemble type");
//        }
//    }
//
//    public kNN getBuiltClassifier(ClassifierVariants classifierType) throws Exception{
//        if(!this.classifierBuilt){
//            throw new Exception("Error: this ensemble has not been built yet");
//        }
//        for(int c = 0; c < this.finalClassifierTypes.length; c++){
//            if(this.finalClassifierTypes[c]==classifierType){
//                return this.builtClassifiers[c];
//            }
//        }
//        throw new Exception("Error: "+classifierType+" hasn't been included in this ensemble");
//    }
//
//
//    public static void clusterMaster(String[] args) throws Exception{
//
//
//        // after 2.0 reboot
//        if(args[0].equalsIgnoreCase("estimateExperimentTimes")){
//            String classifier = args[1];
//            int dataId = Integer.parseInt(args[2])-1;
//            runClusterTimeEstimations(classifier, dataId);
//        }else if(args[0].equalsIgnoreCase("parseExperimentTimes")){
//            parseTimeEstimationsOnCluster();
//        }else if(args[0].equalsIgnoreCase("writeLOOCVScripts")){
//            writeScripts_allParams_instanceResampling_LOOCVOnly();
//        }else if(args[0].equalsIgnoreCase("cvAllParamsInstanceResampling")){
//
//            String datasetName = args[1];
//            int resampleSeed = Integer.parseInt(args[2]);
//            String instancesAddress = args[3];
//            int foldSize = Integer.parseInt(args[4]);
//            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[5]);
//
//            String fullDataPath = instancesAddress+datasetName+"/"+datasetName+"_";
//            Instances originalTrain = ClassifierTools.loadData(fullDataPath+"TRAIN");
//            Instances originalTest = ClassifierTools.loadData(fullDataPath+"TRAIN");
//
//            Instances train = InstanceTools.resampleTrainAndTestInstances(originalTrain, originalTest, resampleSeed)[0];
////            for(int i = 0; i < train.numInstances(); i++){
//                ElasticEnsembleCluster.writeCvVariableFoldSizeAllParamsResultsForCluster(train, datasetName, datasetName+"_"+resampleSeed, classifier, foldSize);
////            }
//        }else if(args[0].equalsIgnoreCase("parseResultsCluster")){
//
////            ClassifierVariants[] classifiers = {ClassifierVariants.Euclidean_1NN, ClassifierVariants.DTW_R1_1NN, ClassifierVariants.DTW_Rn_1NN};
//            ClassifierVariants[] classifiers = ClassifierVariants.values();
////            String[] datasets = {"Beef"};
////            String[] datasets = {"BeetleFly"};
//
//            FileWriter parseLog = new FileWriter("parseLog.txt");
//            parseLog.close();
//            for(String dataset:datasets){
////                if(dataset.equalsIgnoreCase("Adiac") || dataset.equalsIgnoreCase("Beef") || dataset.equalsIgnoreCase("BeetleFly")){
////                    continue;
////                }
//                for(ClassifierVariants classifier:classifiers){
//                    for(int i = 1; i <=100; i++){
//                        try{
//                            writeBestParamFileFromClusterOutput(dataset, dataset+"_"+i, classifier, true);
//                        }catch(Exception e){
////                            parseLog = new FileWriter("parseLog.txt",true);
////                            parseLog.append(dataset+"_"+i+"_"+classifier+"\n");
////                            parseLog.close();
//                        }
//                    }
//                }
//            }
////
//        }else if(args[0].equalsIgnoreCase("parseResultsClusterInTens")){
//
////            ClassifierVariants[] classifiers = {ClassifierVariants.Euclidean_1NN, ClassifierVariants.DTW_R1_1NN, ClassifierVariants.DTW_Rn_1NN};
//            ClassifierVariants[] classifiers = ClassifierVariants.values();
////            String[] datasets = {"Beef"};
////            String[] datasets = {"BeetleFly"};
//
//            int datasetOffset = Integer.parseInt(args[1])-1;
//
//            FileWriter parseLog = new FileWriter("parseLog.txt");
//            parseLog.close();
//            String dataset;
//            for(int d = datasetOffset*10 ; d < datasetOffset*10+10; d++){
//                if(d < datasets.length){
//                    dataset = datasets[d];
//                }else{
//                    return;
//                }
////                if(dataset.equalsIgnoreCase("Adiac") || dataset.equalsIgnoreCase("Beef") || dataset.equalsIgnoreCase("BeetleFly")){
////                    continue;
////                }
//                for(ClassifierVariants classifier:classifiers){
//                    for(int i = 1; i <=100; i++){
//                        try{
//                            writeBestParamFileFromClusterOutput(dataset, dataset+"_"+i, classifier, true);
//                        }catch(Exception e){
////                            System.out.println(e);
//                            e.printStackTrace();
////                            parseLog = new FileWriter("parseLog.txt",true);
////                            parseLog.append(dataset+"_"+i+"_"+classifier+"\n");
////                            parseLog.close();
//                        }
//                    }
//                }
//            }
////
//        }else if(args[0].equalsIgnoreCase("parseInDetail")){
//
//            String datasetName = args[1];
//            String resampleId = args[2];
//            try{
////                writeBestParamFileFromClusterOutput("Worms", "Worms_90", ClassifierVariants.MSM_1NN, true);
//                writeBestParamFileFromClusterOutput(datasetName, datasetName+"_"+resampleId, ClassifierVariants.MSM_1NN, true);
//                System.out.println("Done (apparently)");
//            }catch(Exception e){
//                            System.out.println(e);
//                e.printStackTrace();
////                            parseLog = new FileWriter("parseLog.txt",true);
////                            parseLog.append(dataset+"_"+i+"_"+classifier+"\n");
////                            parseLog.close();
//            }
//
//        }else if(args[0].equalsIgnoreCase("finishedDatasetsReport")){
//            reportOnFinishedClassifiersAndDatasets();
//        }else if(args[0].equalsIgnoreCase("writeTestScripts")){
//            writeScripts_testClassification_individualClassifiers();
//        }else if(args[0].equalsIgnoreCase("testClassification")){
//            String datasetName = args[1];
//            int resampleSeed = Integer.parseInt(args[2]);
//            String instancesAddress = args[3];
//            int foldSize = Integer.parseInt(args[4]);
//            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[5]);
//
//
//
//            try{
//                clusterTestClassification(datasetName, classifier, resampleSeed);
//            }catch(Exception e){
//                // do nothing, as i don't want the output files being full of text!
//            }
//        }else if(args[0].equalsIgnoreCase("makeFullPropScripts")){
//            writeScripts_fullPropEE();
//        }else if(args[0].equalsIgnoreCase("fullPropEE")){
//
//            // for now, include all classifiers from the ensemble. Later, we can look at trying all combinations (for information, as it's completely biased)
//
//
//            String dataset = args[1].trim();
//            int resampleId = Integer.parseInt(args[2].trim());
//
//            clusterEETestClassification(dataset, resampleId);
//        }else if(args[0].equalsIgnoreCase("parseEETestByFold")){
//            parseClusterEETestResults();
//        }else{
//            System.out.println("You shouldn't be here?!");
//        }
//
//
//
////        if(args[0].equalsIgnoreCase("resampleScriptMaker")){
////            writeScripts_allParams_instanceResampling();
////        }else if(args[0].equalsIgnoreCase("start")){
//////            clusterMaster(args[1], args[2]);
////        }else if(args[0].equalsIgnoreCase("cvClassification")){
////
////            String outputIdentifier = args[1];
////            String instancesAddress = args[2];
////            int paramId = Integer.parseInt(args[3].trim())-1;
////            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[4]);
////
////            Instances train = ClassifierTools.loadData(instancesAddress);
////            for(int i = 0; i < train.numInstances(); i++){
////                ElasticEnsembleCluster.writeCvResultsForCluster(train, outputIdentifier, classifier, i, paramId);
////            }
////        }else if(args[0].equalsIgnoreCase("cvClassificationAllParam")){
////
////            String outputIdentifier = args[1];
////            String instancesAddress = args[2];
////            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[3]);
////
////            Instances train = ClassifierTools.loadData(instancesAddress);
////            for(int i = 0; i < train.numInstances(); i++){
////                // needs to be ammended because we now pass the dataset name too
//////                ElasticEnsembleCluster.writeCvVariableFoldSizeAllParamsResultsForCluster(train, outputIdentifier, classifier, 1);
////            }
////        }else if(args[0].equalsIgnoreCase("cvAllParamsInstanceResampling")){
////
////            String datasetName = args[1];
////            int resampleSeed = Integer.parseInt(args[2]);
////            String instancesAddress = args[3];
////            int foldSize = Integer.parseInt(args[4]);
////            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[5]);
////
////            String fullDataPath = instancesAddress+datasetName+"/"+datasetName+"_";
////            Instances originalTrain = ClassifierTools.loadData(fullDataPath+"TRAIN");
////            Instances originalTest = ClassifierTools.loadData(fullDataPath+"TRAIN");
////
////            Instances train = InstanceTools.resampleTrainAndTestInstances(originalTrain, originalTest, resampleSeed)[0];
//////            for(int i = 0; i < train.numInstances(); i++){
////                ElasticEnsembleCluster.writeCvVariableFoldSizeAllParamsResultsForCluster(train, datasetName, datasetName+"_"+resampleSeed, classifier, foldSize);
//////            }
////        }else if(args[0].equalsIgnoreCase("parseResults")){
////
////            String outputIdentifier = args[1];
////            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[2]);
////            ElasticEnsembleCluster.writeBestParamFile(outputIdentifier, classifier, true);
////
//////        }else if(args[0].equalsIgnoreCase("parseResultsCluster")){
//////            String dataName = args[1];
//////            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[2]);
//////
//////            for(int i = 1; i <= 100; i++){
//////                writeBestParamFileFromClusterOutput(dataName, dataName+"_"+i, classifier, true);
//////            }
////        }else if(args[0].equalsIgnoreCase("parseResultsCluster")){
////
//////            String[] datasets = {"MoteStrain","SonyAIBORobotSurface"};
////            ClassifierVariants[] classifiers = {ClassifierVariants.Euclidean_1NN, ClassifierVariants.DTW_R1_1NN, ClassifierVariants.DTW_Rn_1NN};
////
//////            String dataName = args[1];
//////            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[2]);
////
////            FileWriter parseLog = new FileWriter("parseLog.txt");
////            parseLog.close();
////            for(String dataset:datasets){
////                for(ClassifierVariants classifier:classifiers){
////                    for(int i = 1; i <=100; i++){
////                        try{
////                            writeBestParamFileFromClusterOutput(dataset, dataset+"_"+i, classifier, false);
////                        }catch(Exception e){
////                            parseLog = new FileWriter("parseLog.txt",true);
////                            parseLog.append(dataset+"_"+i+"_"+classifier+"\n");
////                            parseLog.close();
////                        }
////                    }
////                }
////            }
////
////        }else if(args[0].equalsIgnoreCase("missingExperimentFinder")){
////            ClassifierVariants[] toLookFor = {ClassifierVariants.Euclidean_1NN, ClassifierVariants.DTW_R1_1NN, ClassifierVariants.DTW_Rn_1NN};
////            findMissingResultsCluster(datasets, toLookFor);
////        }else if(args[0].equalsIgnoreCase("missingExperimentFinderTestRun")){
////            ClassifierVariants[] toLookFor = {ClassifierVariants.Euclidean_1NN, ClassifierVariants.DTW_R1_1NN, ClassifierVariants.DTW_Rn_1NN};
////            findMissingResultsCluster(new String[]{"MoteStrain","SonyAIBORobotSurface"}, toLookFor);
////        }else if(args[0].equalsIgnoreCase("buildClassifierFromParsedTest")){
//////            String[] datasets = {"MoteStrain","SonyAIBORobotSurface"};
//////            ClassifierVariants[] toLookFor = {ClassifierVariants.Euclidean_1NN, ClassifierVariants.DTW_R1_1NN, ClassifierVariants.DTW_Rn_1NN};
////
////        }else if(args[0].equalsIgnoreCase("estimateExperimentTimes")){
////            String classifier = args[1];
////            int dataId = Integer.parseInt(args[2])-1;
////            runClusterTimeEstimations(classifier, dataId);
////        }else if(args[0].equalsIgnoreCase("parseExperimentTimes")){
////            parseTimeEstimationsOnCluster();
////        }else{
////            System.out.println("You shouldn't be here?!");
////        }
//    }
//
//    public static int estimateLOOCVRuntime(ClassifierVariants classifierType, Instances train) throws Exception{
//        DistanceFunction df;
//        DecimalFormat format;
//
//        int numIterations = 5;
//
//        long start, end, cumulativeTime;
//        int days,hours,mins,secs,rem;
//        int numFolds;
//        long maxTimeByParam = 0;
//
//        for(int p = 0; p < 100; p++){
//            cumulativeTime = 0;
//
//            for(int i = 0; i < numIterations; i++){
//                df = getInternalClassifier(classifierType, 1, train).getNearestNeighbourSearchAlgorithm().getDistanceFunction(); // make sure there's no weird in-memory caching by creating new every time
//                start = System.nanoTime();
//                df.distance(train.instance(0), train.instance(1));
//                end = System.nanoTime();
//                cumulativeTime += end-start;
//            }
//            if(cumulativeTime > maxTimeByParam){
//                maxTimeByParam = cumulativeTime;
//            }
//        }
//
//        // Worst-case LOOCV time is estimated by taking the max time for a ginle distance calculation.
//        // For a single instances to be evaluated, it must be compared to all instances except itself.
//        // This then must be replicated for each instance. Hence:
//        // long estimateLOOCV = maxTimeByParam*(train.numInstances()-1)*train.numInstances();
//
//        long totalTrainingEstimate;
//
//        boolean lessThan24hrs = false;
//        int bsfFoldSize = -1;
//        int bsfNumFolds = -1;
//        for(int foldSize = 1; foldSize < train.numInstances() && !lessThan24hrs; foldSize++){
//            numFolds = (int)Math.ceil(train.numInstances()/foldSize);
//            // singleCalcTime * numNotInFold * numFolds;
//            totalTrainingEstimate = maxTimeByParam*(train.numInstances()-foldSize)*numFolds;
//
//            secs = (int)(totalTrainingEstimate/1000000000);
//
//            days = secs/(60*60*24);
//            rem  = secs%(60*60*24);
//
////            if(days > 1){
////                continue;
////            }else{
////                lessThan24hrs=true;
////            }
////            if(days==0){
////                lessThan24hrs=true;
////                bsfFoldSize = foldSize;
////                bsfNumFolds = numFolds;
////            }
//
//            hours = rem/(60*60);
//            rem = rem%(60*60);
//
//            mins = rem/60;
//            secs = rem%60;
//
//            if(days ==0 && hours<5){
//                lessThan24hrs=true;
//                bsfFoldSize = foldSize;
//                bsfNumFolds = numFolds;
//            }
//
//            format = new DecimalFormat();
//            format.setMinimumIntegerDigits(2);
//            System.out.println("Expected runtime: "+days+" day(s) and "+format.format(hours)+":"+format.format(mins)+":"+format.format(secs));
//        }
//
//        if(!lessThan24hrs){
//            throw new Exception("ERROR: Dataset relation "+train.relationName()+" cannont be processed in under 24 hours");
//
//        }
//        System.out.println(bsfNumFolds+" folds, "+bsfFoldSize+" instances per fold");
//        return bsfFoldSize;
//
//
//    }
//
//
//    public static int estimateLOOCVRuntimeAllParams(ClassifierVariants classifierType, Instances train) throws Exception{
//        DistanceFunction df;
//        DecimalFormat format;
//
//        int numIterations = 10;
//
//        long start, end, runtime, maxRuntime;
//        int days =0;
//        int hours =0;
//        int mins = 0;
//        int secs = 0;
//        int rem;
//        int numFolds;
////        long[] maxTimeByParamInSeconds = new long[100];
//        long[] maxTimeByParam = new long[100];
//
//        Random r = new Random(0);
//        for(int p = 0; p < 100; p++){
//            runtime = 0;
//            maxRuntime = 0;
//            for(int i = 0; i < numIterations; i++){
//                df = getInternalClassifier(classifierType, 1, train).getNearestNeighbourSearchAlgorithm().getDistanceFunction(); // make sure there's no weird in-memory caching by creating new every time
//                start = System.nanoTime();
////                df.distance(train.instance(0), train.instance(1));
//                df.distance(train.instance(r.nextInt(train.numInstances())), train.instance(r.nextInt(train.numInstances())));
//                end = System.nanoTime();
//                runtime = end-start;
////                System.out.println(runtime);
//                if(runtime > maxRuntime){
//                    maxRuntime = runtime;
//                }
//            }
////            maxTimeByParamInSeconds[p] = maxRuntime/1000000000+1; // Add a second to remove rounding error without casting
//            maxTimeByParam[p] = maxRuntime;
////            System.out.println();
//        }
//
//        // Worst-case LOOCV time is estimated by taking the max time for a ginle distance calculation.
//        // For a single instances to be evaluated, it must be compared to all instances except itself.
//        // This then must be replicated for each instance. Hence:
//        // long estimateLOOCV = maxTimeByParam*(train.numInstances()-1)*train.numInstances();
//
//        long totalTrainingEstimate;
//
//        boolean lessThan24hrs = false;
//        int bsfFoldSize = -1;
//        int bsfNumFolds = -1;
//        long totalTimeEstimate;
//        for(int foldSize = 1; foldSize < train.numInstances() && !lessThan24hrs; foldSize++){
//            numFolds = (int)Math.ceil(train.numInstances()/foldSize);
//            // singleCalcTime * numNotInFold * numFolds;
//
//            totalTimeEstimate = 0;
//            for(int p = 0; p < 100; p++){
//                totalTimeEstimate += maxTimeByParam[p]*(train.numInstances()-foldSize)*numFolds;
//            }
//
//
////            totalTrainingEstimate = maxTimeByParam*(train.numInstances()-foldSize)*numFolds;
//
////            secs = (int)(totalTrainingEstimate/1000000000);
//            secs = (int)(totalTimeEstimate/1000000000);
//
//            days = secs/(60*60*24);
//            rem  = secs%(60*60*24);
//
//            hours = rem/(60*60);
//            rem = rem%(60*60);
//
//            mins = rem/60;
//            secs = rem%60;
//
//            if(days < 1){
//                lessThan24hrs=true;
//                bsfFoldSize = foldSize;
//                bsfNumFolds = numFolds;
//            }
//
////            if(days == 0 && hours<5){
////                lessThan24hrs=true;
////                bsfFoldSize = foldSize;
////                bsfNumFolds = numFolds;
////            }
//
////            format = new DecimalFormat();
////            format.setMinimumIntegerDigits(2);
////            System.out.println("Expected runtime: "+days+" day(s) and "+format.format(hours)+":"+format.format(mins)+":"+format.format(secs));
//        }
//
//        if(!lessThan24hrs){
//            throw new Exception("ERROR: Dataset relation "+train.relationName()+" cannont be processed in under 24 hours");
//
//        }
////        System.out.println(bsfNumFolds+" folds, "+bsfFoldSize+" instances per fold");
//
//        format = new DecimalFormat();
//        format.setMinimumIntegerDigits(2);
////            System.out.println("Expected runtime: "+days+" day(s) and "+format.format(hours)+":"+format.format(mins)+":"+format.format(secs));
//        System.out.println(bsfNumFolds+","+bsfFoldSize+","+days+","+format.format(hours)+":"+format.format(mins)+":"+format.format(secs));
//        return bsfFoldSize;
//
//
//    }
//
//    public static void printAllTimeEstimates() throws Exception{
//        System.out.println("dataset,classifier,numFolds,foldSize,days,hours:mins:secs");
//        ClassifierVariants[] classifiers = ClassifierVariants.values();
//        for(int f = 0; f < datasets.length; f++){
//            Instances train = ClassifierTools.loadData("C:/Temp/Dropbox/TSC Problems/"+datasets[f]+"/"+datasets[f]+"_TRAIN");
//
//            for(int c = 0; c < classifiers.length; c++){
//                System.out.print(datasets[f]+","+classifiers[c]+",");
//                estimateLOOCVRuntimeAllParams(classifiers[c], train);
//            }
//
//
//
//        }
//    }
//
//
//
//
//    public static void writeScripts_allParams_instanceResampling() throws Exception{
//        new File("eeClusterScripts/").mkdirs();
//
////        Scanner scan = new Scanner(new File("C:/Jay/2015/EEResampleTimings_adjusted30Mins.csv"));
//        Scanner scan = new Scanner(new File("EEResampleTimings_adjusted30Mins.csv"));
//        scan.useDelimiter("\n");
//        scan.next(); // header
//        String[] line, time;
//        String jobString, jobName ,dataset, classifier, foldSize;
//
//        HashSet<String> toUse = new HashSet<>();
//        toUse.add(ClassifierVariants.Euclidean_1NN.toString());
//        toUse.add(ClassifierVariants.DTW_R1_1NN.toString());
//        toUse.add(ClassifierVariants.DTW_Rn_1NN.toString());
////        toUse.add(ClassifierVariants.WDTW_1NN.toString());
////        toUse.add(ClassifierVariants.DDTW_R1_1NN.toString());
////        toUse.add(ClassifierVariants.DDTW_Rn_1NN.toString());
////        toUse.add(ClassifierVariants.WDDTW_1NN.toString());
////        toUse.add(ClassifierVariants.LCSS_1NN.toString());
////        toUse.add(ClassifierVariants.ERP_1NN.toString());
////        toUse.add(ClassifierVariants.MSM_1NN.toString());
////        toUse.add(ClassifierVariants.TWE_1NN.toString());
//
//
//
//        String part1 = "#!/bin/csh\n\n#BSUB -q short\n#BSUB -J ";
//        String part2 = "[1-100]\n#BSUB -oo output/";
//        String part3 = "%I.out\n#BSUB -eo error/";
//        String part4 = "%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar -Xmx2048m TimeSeriesClassification.jar cvAllParamsInstanceResampling ";
////        String part4 = "%I.err\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar TimeSeriesClassification.jar cvAllParamsInstanceResampling ";
//        String part5 = " $LSB_JOBINDEX \"/gpfs/home/sjx07ngu/TSCProblems/\" ";
//
//
//        FileWriter out;
//        StringBuilder st;
//
//        FileWriter instructionsOut = new FileWriter("instructions.txt");
//
//        while(scan.hasNext()){
//            line = scan.next().split(",");
//            dataset = line[0];
//            classifier = line[1];
//            foldSize = line[3];
//
//            if(toUse.contains(classifier)){
//                new File("eeClusterScripts/"+dataset).mkdir();
//
//                jobName = dataset+"_"+classifier;
//                jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset+part5+foldSize+" "+classifier;
//                out = new FileWriter("eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub");
//                out.append(jobString);
//                out.close();
//
//                time = line[5].split(":");
//                int secs = Integer.parseInt(time[0])*60*60+Integer.parseInt(time[1])*60+Integer.parseInt(time[2]);
//                instructionsOut.append(secs+",bsub < eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub");
//            }
//
//        }
//        instructionsOut.close();
//
//
//
//
//
//    }
//
//    public static void writeScripts_allParams_instanceResampling_LOOCVOnly() throws Exception{
//        new File("eeClusterScripts/").mkdirs();
//
//        HashSet<String> datasetsToAvoid = new HashSet<>();
//        datasetsToAvoid.add("ElectricDevices");
//        datasetsToAvoid.add("FordA");
//        datasetsToAvoid.add("FordB");
//        datasetsToAvoid.add("HandOutlines");
//        datasetsToAvoid.add("LargeKitchenAppliances");
//        datasetsToAvoid.add("NonInvasiveFatalECG_Thorax1");
//        datasetsToAvoid.add("NonInvasiveFatalECG_Thorax2");
//        datasetsToAvoid.add("RefrigerationDevices");
//        datasetsToAvoid.add("ScreenType");
//        datasetsToAvoid.add("ShapesAll");
//        datasetsToAvoid.add("SmallKitchenAppliances");
//        datasetsToAvoid.add("StarLightCurves");
//        datasetsToAvoid.add("UWaveGestureLibrary_X");
//        datasetsToAvoid.add("UWaveGestureLibrary_Y");
//        datasetsToAvoid.add("UWaveGestureLibrary_Z");
//        datasetsToAvoid.add("UWaveGestureLibraryAll");
//
//        ArrayList<ClassifierVariants> classifiersToUse = new ArrayList<>();
//        classifiersToUse.add(ClassifierVariants.Euclidean_1NN);
//        classifiersToUse.add(ClassifierVariants.DTW_R1_1NN);
//        classifiersToUse.add(ClassifierVariants.DTW_Rn_1NN);
//        classifiersToUse.add(ClassifierVariants.WDTW_1NN);
//        classifiersToUse.add(ClassifierVariants.DDTW_R1_1NN);
//        classifiersToUse.add(ClassifierVariants.DDTW_Rn_1NN);
//        classifiersToUse.add(ClassifierVariants.WDDTW_1NN);
//        classifiersToUse.add(ClassifierVariants.LCSS_1NN);
//        classifiersToUse.add(ClassifierVariants.ERP_1NN);
//        classifiersToUse.add(ClassifierVariants.MSM_1NN);
//        classifiersToUse.add(ClassifierVariants.TWE_1NN);
//
//        String part1 = "#!/bin/csh\n\n#BSUB -q short\n#BSUB -J ";
//        String part2 = "[1-100]\n#BSUB -oo output/";
//        String part3 = "%I.out\n#BSUB -eo error/";
//        String part4 = "%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar -Xmx2048m TimeSeriesClassification.jar cvAllParamsInstanceResampling ";
//        String part5 = " $LSB_JOBINDEX \"/gpfs/home/sjx07ngu/TSCProblems/\" ";
//
//
//        FileWriter out;
//        String[] timeLineParts;
//        String jobString, jobName;
//
//
//        FileWriter instructionsOut = new FileWriter("instructions.txt");
//        Scanner scan;
//        for(String dataset:datasets){
//            if(datasetsToAvoid.contains(dataset)){
//                continue; // avoiding datasets with > 20 hour runtime for now
//            }
//
//            // load timing info
//
//            for(ClassifierVariants classifier: classifiersToUse){
//
//
//
//                new File("eeClusterScripts/"+dataset).mkdir();
//
//                jobName = dataset+"_"+classifier;
//                jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset+part5+"1 "+classifier;
//                out = new FileWriter("eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub");
//                out.append(jobString);
//                out.close();
//
//                // load in times
//                scan = new Scanner(new File("../01_timingExperiments/timeEstimations/timeEstimations_"+jobName+".txt"));
//                scan.useDelimiter("\n");
//                timeLineParts = scan.next().split(",");
//
//                if(!timeLineParts[1].trim().equalsIgnoreCase("1")){
//                    throw new Exception("somethings gone wrong here - should only be prepping for LOOCV experiments");
//                }
//
//                instructionsOut.append(timeLineParts[0]+",bsub < eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub\n");
//            }
//        }
//
//        instructionsOut.close();
//
//
//
//
//
//    }
//
//    public static void writeScripts_fullPropEE() throws Exception{
//        new File("eeClusterScripts/").mkdirs();
//
//        HashSet<String> datasetsToAvoid = new HashSet<>();
//        datasetsToAvoid.add("ElectricDevices");
//        datasetsToAvoid.add("FordA");
//        datasetsToAvoid.add("FordB");
//        datasetsToAvoid.add("HandOutlines");
//        datasetsToAvoid.add("LargeKitchenAppliances");
//        datasetsToAvoid.add("NonInvasiveFatalECG_Thorax1");
//        datasetsToAvoid.add("NonInvasiveFatalECG_Thorax2");
//        datasetsToAvoid.add("RefrigerationDevices");
//        datasetsToAvoid.add("ScreenType");
//        datasetsToAvoid.add("ShapesAll");
//        datasetsToAvoid.add("SmallKitchenAppliances");
//        datasetsToAvoid.add("StarLightCurves");
//        datasetsToAvoid.add("UWaveGestureLibrary_X");
//        datasetsToAvoid.add("UWaveGestureLibrary_Y");
//        datasetsToAvoid.add("UWaveGestureLibrary_Z");
//        datasetsToAvoid.add("UWaveGestureLibraryAll");
//
//
//
//        String part1 = "#!/bin/csh\n\n#BSUB -q short\n#BSUB -J ";
//        String part2 = "[1-100]\n#BSUB -oo output/";
//        String part3 = "%I.out\n#BSUB -eo error/";
//        String part4 = "%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar -Xmx2048m TimeSeriesClassification.jar fullPropEE ";
//        String part5 = " $LSB_JOBINDEX ";
//
//
//        FileWriter out;
//        String[] timeLineParts;
//        String jobString, jobName;
//
//
//        FileWriter instructionsOut = new FileWriter("instructions.txt");
//        Scanner scan;
//        for(String dataset:datasets){
//            if(datasetsToAvoid.contains(dataset)){
//                continue; // avoiding datasets with > 20 hour runtime for now
//            }
//
//            new File("eeClusterScripts/").mkdir();
//
//            jobName = "fullPropEE_"+dataset;
//            jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset+part5;
//            out = new FileWriter("eeClusterScripts/"+dataset+".bsub");
//            out.append(jobString);
//            out.close();
//
//            instructionsOut.append("bsub < eeClusterScripts/"+dataset+".bsub\n");
//
//        }
//
//        instructionsOut.close();
//
//    }
//
//    public static void findMissingResultsCluster(){
//        findMissingResultsCluster(datasets, ClassifierVariants.values());
//    }
//
//    public static void findMissingResultsCluster(String[] datasetNames, ClassifierVariants[] classifiers){
//
////        ClassifierVariants[] classifiers = ClassifierVariants.values();
//        int[] params = new int[classifiers.length];
//        for(int c = 0; c < classifiers.length; c++){
//            if(classifiers[c].equals(ClassifierVariants.Euclidean_1NN) || classifiers[c].equals(ClassifierVariants.DTW_R1_1NN) || classifiers[c].equals(ClassifierVariants.DDTW_R1_1NN)){
//                params[c] = 1;
//            }else{
//                params[c] = 100;
//            }
//        }
//
//        String dataDir, resampleDir, classifierDir, fileLoc;
//        File individualResult;
//
//        ArrayList<String> missingDatasets = new ArrayList<>();
//        HashMap<String, ArrayList<Integer>> missingResamples = new HashMap<>();
//        HashMap<String, ArrayList<ClassifierVariants>> missingResampleClassifiers = new HashMap<>();
//        HashMap<String,  ArrayList<Integer>> missingResampleClassifierParamExperimets = new HashMap<>();
//
//
//        for(String dataset:datasetNames){
//            if(!new File("eeClusterOutput/"+dataset).exists()){
//                missingDatasets.add(dataset);
//                continue;
//            }
//
//            for(int i = 1; i <= 100; i++){
//
//                if(!new File("eeClusterOutput/"+dataset+"/"+dataset+"_"+i).exists()){
//
//                    if(!missingResamples.containsKey(dataset)){
//                        missingResamples.put(dataset, new ArrayList<Integer>());
//                    }
//                    missingResamples.get(dataset).add(i);
//                    continue;
//                }
//
//
//
//                for(int c = 0; c < classifiers.length; c++){
//
//                    // check whether this dataset/classifier has already been parsed
//                    if(new File("eeClusterOutput/"+dataset+"/"+dataset+"_"+i+"/"+dataset+"_"+i+"_parsedOutput/"+dataset+"_"+i+"_"+classifiers[c].toString()+".txt").exists()){
//                        continue;
//                    }
//
//                    if(!new File("eeClusterOutput/"+dataset+"/"+dataset+"_"+i+"/"+dataset+"_"+i+"_"+classifiers[c].toString()).exists()){
//
//                        if(!missingResampleClassifiers.containsKey(dataset+"_"+i)){
//                            missingResampleClassifiers.put(dataset+"_"+i, new ArrayList<ClassifierVariants>());
//                        }
//                        missingResampleClassifiers.get(dataset+"_"+i).add(classifiers[c]);
//                        continue;
//                    }
//
//                    for(int p = 0; p < params[c]; p++){
////                        fileLoc = classifierDir+"/"+dataset+"_"+i+"_"+classifiers[c].toString()+"_p_"+p+".txt";
//                        individualResult = new File("eeClusterOutput/"+dataset+"/"+dataset+"_"+i+"/"+dataset+"_"+i+"_"+classifiers[c].toString()+"/"+dataset+"_"+i+"_"+classifiers[c].toString()+"_p_"+p+".txt");
//                        if(!individualResult.exists()){
////                            System.out.println(fileLoc);
//                            if(!missingResampleClassifierParamExperimets.containsKey(dataset+"_"+i+"_"+classifiers[c].toString())){
//                                missingResampleClassifierParamExperimets.put(dataset+"_"+i+"_"+classifiers[c].toString(), new ArrayList<Integer>());
//                            }
//                            missingResampleClassifierParamExperimets.get(dataset+"_"+i+"_"+classifiers[c].toString()).add(p);
//
//                        }
//                    }
//
//
//                }
//
//
//            }
//
//        }
//
////        ArrayList<String> missingDatasets = new ArrayList<>();
////        HashMap<String, ArrayList<Integer>> missingResamples = new HashMap<>();
////        HashMap<String, ArrayList<ClassifierVariants>> missingResampleClassifiers = new HashMap<>();
////        HashMap<String,  ArrayList<Integer>> missingResampleClassifierParamExperimets = new HashMap<>();
//        System.out.println("Missing Datasets");
//        for(String d:missingDatasets){
//            System.out.println(d);
//        }
//        System.out.println("\nMissing Resamples:");
//        for(String d:missingResamples.keySet()){
//            System.out.println(d);
//            for(Integer i: missingResamples.get(d)){
//                System.out.println("\t"+i);
//            }
//        }
//        System.out.println("\nMissing Resample Classifiers:");
//        for(String d:missingResampleClassifiers.keySet()){
//            System.out.println(d);
//            for(ClassifierVariants c: missingResampleClassifiers.get(d)){
//                System.out.println("\t"+c);
//            }
//        }
//        System.out.println("\nMissing Param Experiments:");
//        for(String d:missingResampleClassifierParamExperimets.keySet()){
//            System.out.println(d);
//            for(Integer i: missingResampleClassifierParamExperimets.get(d)){
//                System.out.println("\t"+i);
//            }
//        }
//
//
//    }
//
//    public static void buildFromParsedResults_proofOfConcept() throws Exception{
//        String fullDataPath = "C:/Temp/Dropbox/TSC Problems/";
//        String dataset = "MoteStrain";
//        ElasticEnsembleCluster ee;
//        Instances originalTrain = ClassifierTools.loadData(fullDataPath+dataset+"/"+dataset+"_TRAIN");
//        Instances originalTest = ClassifierTools.loadData(fullDataPath+dataset+"/"+dataset+"_TEST");
//
//        Instances[] trainAndTest;
//
//        double[] accuraciesByResample = new double[100];
//        int correct;
//
//        for(int resampleSeed = 1; resampleSeed <= 100; resampleSeed++){
//            ee = new ElasticEnsembleCluster(dataset, resampleSeed);
//            ee.removeAllClassifiersFromEnsemble();
////            ee.addClassifierToEnsemble(ClassifierVariants.Euclidean_1NN);
//            ee.addClassifierToEnsemble(ClassifierVariants.DTW_Rn_1NN);
//
//            trainAndTest = InstanceTools.resampleTrainAndTestInstances(originalTrain, originalTest, resampleSeed);
//            ee.buildClassifier(trainAndTest[0]);
//
//            correct = 0;
//            for(int i = 0; i < trainAndTest[1].numInstances(); i++){
//                if(trainAndTest[1].instance(i).classValue()==ee.classifyInstance(trainAndTest[1].instance(i))){
//                    correct++;
//                }
//            }
//            accuraciesByResample[resampleSeed-1] = (double)correct/trainAndTest[1].numInstances();
//            System.out.println(resampleSeed+" "+accuraciesByResample[resampleSeed-1]);
//        }
//    }
//
//    public static String singleTimingRun(String datasetName, String dataDir, ClassifierVariants classifier, int paramId, int limitInSecs) throws Exception{
////        String clusterDataDir = "/gpfs/home/sjx07ngu/TSCProblems/";
////        String clusterDataDir = "C:/Temp/Dropbox/TSC Problems/";
//        Instances train = ClassifierTools.loadData(dataDir+datasetName+"/"+datasetName+"_TRAIN");
//        Instances test = ClassifierTools.loadData(dataDir+datasetName+"/"+datasetName+"_TEST");
//
//        new File("timeEstimationLogs").mkdir();
//
//        long start, end;
//        kNN oneNN;
//
//        long nanosPerIns;
//        long maxNanosPerIns = -1;
//
//        long worstCaseNanosPerExperiment = -1;
////        long worstCaseSecsPerExperiment = -1;
//
//        boolean belowLimit = false;
//        Random r;
//        int[] randoms;
//
//        int numParams = 100;
//        if(classifier==ClassifierVariants.Euclidean_1NN || classifier == ClassifierVariants.DTW_R1_1NN || classifier == ClassifierVariants.DDTW_R1_1NN){
//            numParams = 1;
//        }
//        // single run to force any unfair (/unknown) computational overheads
//        oneNN = getInternalClassifier(classifier, paramId, train);
//        oneNN.classifyInstance(test.get(0));
//
//        // rather than surveying all possible params, try the highest, (e.g. full window DTW, lowest (0 weight for WDTW), and maybe another
//        Instances foldTrain, foldTest;
//        int outFoldSize = -1;
//        FileWriter out = new FileWriter("timeEstimationLogs/timeEstimationLog_"+datasetName+"_"+classifier.toString()+".txt");
//        out.close();
////        for(int foldSize = 1; !belowLimit && foldSize < train.numInstances(); foldSize++){
//        for(int foldSize = 1; !belowLimit && foldSize < train.numInstances(); foldSize++){
//
//            // save time - after 10, go up in increments of 5
//            if(foldSize > 10){
//                foldSize+=4;
//            }
//
//            maxNanosPerIns = 0;
//            foldTrain = new Instances(train);
//            foldTest = new Instances(train,0);
//
//            for(int f = 0; f < foldSize; f++){
//                foldTest.add(foldTrain.remove(0));
//            }
//
//            oneNN = getInternalClassifier(classifier, paramId, foldTrain);
//            start = System.nanoTime();
//
//            // make sure random instances are selected without impacting time calculation
//            r = new Random();
//            randoms = new int[]{r.nextInt(foldTest.size()),r.nextInt(foldTest.size()),r.nextInt(foldTest.size())};
//
//            for(int j = 0; j < 3; j++){
////                System.out.println("j "+j);
//                oneNN.classifyInstance(foldTest.get(randoms[j]));
//            }
//            end = System.nanoTime();
//            nanosPerIns = (end-start)/3;
////            System.out.print(nanosPerIns);
//            if(nanosPerIns > maxNanosPerIns){
//                maxNanosPerIns = nanosPerIns;
//            }
////            System.out.println("  "+maxNanosPerIns);
//
//            worstCaseNanosPerExperiment = maxNanosPerIns*train.numInstances()*numParams;
//
//            out = new FileWriter("timeEstimationLogs/timeEstimationLog_"+datasetName+"_"+classifier.toString()+".txt", true);
//            out.append(foldSize+","+worstCaseNanosPerExperiment+","+(worstCaseNanosPerExperiment/1000000000)+"\n");
//            out.close();
////            System.out.println(worstCaseNanosPerExperiment);"timeEstimations/timeEstimations_"+dataName+"_"+classifier.toString()+".txt"
//
////            System.out.println("worst case: "+worstCaseNanosPerExperiment);
//            if(worstCaseNanosPerExperiment/1000000000 <= limitInSecs){
//                belowLimit=true;
//                outFoldSize = foldSize;
//            }
//        }
//
//        return (worstCaseNanosPerExperiment/1000000000)+","+outFoldSize;
//
////        worstCaseSecsPerExperiment = worstCaseSecsPerExperiment/1000000000;
////        System.out.println(worstCaseSecsPerExperiment);
////        System.out.println(worstCaseSecsPerExperiment/60);
////        System.out.println(worstCaseSecsPerExperiment/60/60);
////
////        return null;
//    }
//
//    // specialised code for each classifier variant - eg full window DTW is going to take longer than 1%, so why bother with lower
//    // conversely, WDTW is slowest when weight is 0 (equiv. to full DTW)
//    public static String clusterTimeEstimation(String datasetName, ClassifierVariants classifier) throws Exception{
//
//        String clusterDataDir = "/gpfs/home/sjx07ngu/TSCProblems/";
////        String clusterDataDir = "C:/Temp/Dropbox/TSC Problems/";
//
//
//        int limitInSecs = 60*60*20; // 20 hours
////        limitInSecs = 1;
//        String result = null;
//        switch(classifier){
//            case Euclidean_1NN:
//            case DTW_R1_1NN:
//            case DDTW_R1_1NN:
//            case WDTW_1NN:
//            case WDDTW_1NN:
//            case MSM_1NN: // not sure this will have much effect, smaller cost likely to force more comparisons
//                result = singleTimingRun(datasetName, clusterDataDir, classifier, 0, limitInSecs);
//                break;
//            case DTW_Rn_1NN:
//            case DDTW_Rn_1NN:
//            case ERP_1NN: // bigger band
//            case TWE_1NN: // think best case == worst case == average case on this one
//            case LCSS_1NN: // bigger delta == bigger band
//                result = singleTimingRun(datasetName, clusterDataDir, classifier, 99, limitInSecs);
//                break;
//            default:
//                throw new Exception("Unsupported classifier: "+classifier.toString());
//        }
////        System.out.println(result);
//        return result;
////
////
////
////
////        if(classifier == ClassifierVariants.Euclidean_1NN || classifier == ClassifierVariants.DTW_R1_1NN || classifier == ClassifierVariants.DDTW_R1_1NN){
////            numParams = 1;
////        }
////        int[] secPerIns = new int[numParams];
////
////        kNN oneNN;
////
////        long nanosPerIns;
////        long maxNanosPerIns = -1;
////
////
////        long worstCaseNanosPerExperiment = -1;
////        long worstCaseSecsPerExperiment = -1;
////
////        int limit = 60*60*20; // 20 hours
////
////        boolean belowLimit = false;
////
////        // single run to force any unfair (/unknown) computational overheads
////        oneNN = getInternalClassifier(classifier, 1, train);
////        oneNN.classifyInstance(test.get(0));
////
////        // rather than surveying all possible params, try the highest, (e.g. full window DTW, lowest (0 weight for WDTW), and maybe another
////        Instances foldTrain, foldTest;
////        for(int foldSize = 1; !belowLimit && foldSize < train.numInstances(); foldSize++){
////
////            foldTrain = new Instances(train);
////            foldTest = new Instances(train,0);
////
////            for(int f = 0; f < foldSize; f++){
////                foldTest.add(foldTrain.remove(f));
////            }
////
////            secPerIns = new int[numParams];
////
//////            for(int i = 0; i < numParams; i++){ // don'
////            int i = 100;
////                oneNN = getInternalClassifier(classifier, i, train);
////                start = System.nanoTime();
////                for(int j = 0; j < 5; j++){
////                    System.out.println("j "+j);
////                    oneNN.classifyInstance(test.get(j));
////                }
////                end = System.nanoTime();
////                nanosPerIns = (end-start)/5;
////                System.out.print(nanosPerIns);
////                if(nanosPerIns > maxNanosPerIns){
////                    maxNanosPerIns = nanosPerIns;
////                }
////                System.out.println("  "+maxNanosPerIns);
//////            }
////            worstCaseNanosPerExperiment = maxNanosPerIns*train.numInstances();
////            System.out.println("worst case: "+worstCaseNanosPerExperiment);
////            if(worstCaseSecsPerExperiment/1000000000 <= limit){
////                belowLimit=true;
////            }
////
////        }
////        worstCaseSecsPerExperiment = worstCaseSecsPerExperiment/1000000000;
////        System.out.println(worstCaseSecsPerExperiment);
////        System.out.println(worstCaseSecsPerExperiment/60);
////        System.out.println(worstCaseSecsPerExperiment/60/60);
//
//    }
//
//
//
//
//    /**
//     * Main method
//     *
//     * @param args String[] Arguments used to invoke the main logic of the program.
//     * @throws Exception
//     */
//    public static void main(String[] args) throws Exception{
////        clusterMaster(args);
////        String dataset = "ItalyPowerDemand";
////        String dataset = "StarlightCurves";
////        String dataset = "HandOutlines";
////        String dataset = "FordB";
////        String dataset = "NonInvasiveFatalECG_Thorax1";
////        estimateLOOCVRuntime("C:/Temp/Dropbox/TSC Problems/"+dataset+"/"+dataset+"_TRAIN", ElasticEnsemble.ClassifierVariants.WDTW_1NN);
//
////        Instances train = ClassifierTools.loadData("C:/Temp/Dropbox/TSC Problems/"+dataset+"/"+dataset+"_TRAIN");
////        estimateLOOCVRuntime(ElasticEnsemble.ClassifierVariants.WDTW_1NN, train);
//
////        writeCvVariableFoldSizeAllParamsResultsForCluster(train, "newTestWithItalyCounting", ElasticEnsemble.ClassifierVariants.DTW_R1_1NN, 1);
////        writeCvVariableFoldSizeAllParamsResultsForCluster(train, "newTestWithItalyCounting2", ElasticEnsemble.ClassifierVariants.DTW_R1_1NN, 2);
////        writeCvVariableFoldSizeAllParamsResultsForCluster(train, "newTestWithItaly", ElasticEnsemble.ClassifierVariants.DTW_Rn_1NN, 1);
//
////        estimateLOOCVRuntimeAllParams(ElasticEnsemble.ClassifierVariants.DDTW_Rn_1NN, train);
//
////        printAllTimeEstimates();
//
////        for(int i = 0; i < datasets.length; i++){
////            System.out.print(datasets[i]+" ");
////        }
//
////        writeScripts_allParams_instanceResampling();
//
////        clusterMaster(args);
////        distributionForinstanceTest();
//
////        String[] datasets = {"MoteStrain","SonyAIBORobotSurface"};
////        String[] datasets = {"Beef"};
////
////        ClassifierVariants[] classifiers = {ClassifierVariants.Euclidean_1NN, ClassifierVariants.DTW_R1_1NN, ClassifierVariants.DTW_Rn_1NN};
////
////        for(String dataset:datasets){
////            tidyUpIfParsedExistsCluster(dataset,classifiers);
////        }
////        buildFromParsedResults_proofOfConcept();
//
////        clusterTimeEstimation("ItalyPowerDemand", ElasticEnsemble.ClassifierVariants.DTW_Rn_1NN);
////        clusterTimeEstimation("StarlightCurves", ElasticEnsemble.ClassifierVariants.DTW_Rn_1NN);
//
////        clusterTimeEstimation("FordA", ElasticEnsemble.ClassifierVariants.TWE_1NN);
////        for(int i = 0; i < datasets.length*11;i++){
////            runClusterTimeEstimations(i);
////        }
//
////        clusterTimeEstimation("Worms", ElasticEnsemble.ClassifierVariants.WDTW_1NN);
//        clusterMaster(args);
//
////        printTimeScripts();
//    }
//
//
//    public static void runClusterTimeEstimations(int jobId) throws Exception{
//
//        File outDir = new File("timeEstimations");
//        outDir.mkdirs();
//
//        String dataName = datasets[jobId/11];
//        ClassifierVariants classifier = ClassifierVariants.values()[jobId%11];
////        System.out.println(dataName+" "+classifier);
//
//        FileWriter out = new FileWriter("timeEstimations/timeEstimations_"+dataName+"_"+classifier.toString()+".txt");
//        out.append(clusterTimeEstimation(dataName, classifier));
//        out.close();
//    }
//
//    public static void runClusterTimeEstimations(String classifierName, int dataId) throws Exception{
//
//        File outDir = new File("timeEstimations");
//        outDir.mkdirs();
//
//        String dataName = datasets[dataId];
//
////        if(!dataName.equalsIgnoreCase("UWaveGestureLibrary_X") &&!dataName.equalsIgnoreCase("UWaveGestureLibrary_Y") && !dataName.equalsIgnoreCase("UWaveGestureLibrary_Z") && !dataName.equalsIgnoreCase("StarLightCurves")){
//        if(!dataName.equalsIgnoreCase("Adiac")){
//            return;
//        }
//
//        ClassifierVariants classifier = ClassifierVariants.valueOf(classifierName);
////        System.out.println(dataName+" "+classifier);
//
//        FileWriter out = new FileWriter("timeEstimations/timeEstimations_"+dataName+"_"+classifier.toString()+".txt");
//        out.append(clusterTimeEstimation(dataName, classifier));
//        out.close();
//    }
//
//    public static void printTimeScripts() throws Exception{
//
//        ClassifierVariants[] classifiers = ClassifierVariants.values();
//        new File("timeScripts").mkdir();
//        FileWriter out;
//
//        FileWriter instructionsOut = new FileWriter("timeScripts_instructions.txt");
//        for(ClassifierVariants classifier:classifiers){
//            out = new FileWriter("timeScripts/timings_"+classifier+".bsub");
//            out.append("#!/bin/csh\n\nmkdir -p output\nmkdir -p error\n#BSUB -q short\n#BSUB -J times"+classifier+"[1-81]\n#BSUB -oo output/times"+classifier+"I.out\n#BSUB -eo error/times"+classifier+"%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar TimeSeriesClassification.jar estimateExperimentTimes "+classifier+" $LSB_JOBINDEX");
//            out.close();
//            instructionsOut.append("bsub < timeScripts/timings_"+classifier+".bsub\n");
//        }
//        instructionsOut.close();
//
//
//
//    }
//
//    public static void parseTimeEstimationsOnCluster() throws Exception{
//
//        File[] list = new File("timeEstimations").listFiles();
//        Scanner scan;
//        String[] fileParts;
//        int numFolds;
//        long nanosPerExperiment;
//
//        TreeMap<Integer, ArrayList<String>> listByFolds = new TreeMap<>();
//        ArrayList<String> temp;
//        String expId;
//        for(File file:list){
//            scan = new Scanner(file);
//            scan.useDelimiter("\n");
//            fileParts = scan.next().split(",");
//            nanosPerExperiment = Long.parseLong(fileParts[0]);
//            numFolds = Integer.parseInt(fileParts[1]);
//            expId = file.getName().replace("timeEstimations_", "").replace(".txt", "");
//            if(!listByFolds.containsKey(numFolds)){
//                temp = new ArrayList<>();
//                temp.add(expId);
//                listByFolds.put(numFolds, temp);
//            }else{
//                listByFolds.get(numFolds).add(expId);
//            }
//        }
//
//        for(int foldSize:listByFolds.descendingKeySet()){
//            System.out.println(foldSize+"\n============");
//            for(String id:listByFolds.get(foldSize)){
//                System.out.println(id);
//            }
//        }
//
//    }
//
//    public static void parseClusterTestResults() throws Exception{
//        String clusterDir = "/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/03_classification_loocvDatasets_testResults/eeClusterOutput_testResults/";
//
//        // we want results for a dataset/classifier where 100 runs have been completed. Average accuracy and stdev
//        //row: dataset; column: classifier - avgAcc, stdvAcc;
//        ElasticEnsembleCluster.ClassifierVariants[] classifiers = ElasticEnsembleCluster.ClassifierVariants.values();
//
//        StringBuilder output = new StringBuilder();
//        output.append(",");
//        for(ElasticEnsembleCluster.ClassifierVariants classifier:classifiers){
//            output.append(classifier+",,");
//        }
//        output.delete(output.length()-2, output.length());
//        output.append("\n");
//
//        output.append("Dataset");
//        for(ElasticEnsembleCluster.ClassifierVariants classifier:classifiers){
//            output.append(",acc,stdev");
//        }
//        output.append("\n");
//
//        double[] accs;
//        File result;
//        Scanner scan;
//        boolean writeResult;
//        double sum;
//        double avg;
//        double stdev;
//        for(String dataset:DataSets.fileNames){
//            output.append(dataset);
//            for(ElasticEnsembleCluster.ClassifierVariants classifier:classifiers){
//                writeResult = true;
//                accs = new double[100];
//                sum = 0;
//                // only write to output if all 100 exist
//                for(int f = 1; f <=100 && writeResult; f++){
//
//                    result = new File(clusterDir+dataset+"/"+dataset+"_"+f+"/"+dataset+"_"+f+"_"+classifier+".txt");
//                    if(!result.exists()){
//                        writeResult=false;
//                        continue;
//                    }
//                    // file has accuracy (0-1) on the first row, every subsequent row is prediction/actual[prob_c1, prob_c2,...,prob_cn]
//                    // we don't care about individual predictions at the minute, so just store the double in the first row
////                    scan = new Scanner()
//                    scan = new Scanner(result);
//                    scan.useDelimiter("\n");
//                    accs[f-1] = Double.parseDouble(scan.next().trim());
//                    sum+=accs[f-1];
//                    scan.close();
//                }
//                if(writeResult){
//                    avg = sum/100;
//                    stdev = 0;
//                    for(int i = 0; i < accs.length; i++){
//                        stdev += (accs[i]-avg)*(accs[i]-avg);
//                    }
//                    stdev/=100;
//                    stdev = Math.sqrt(stdev);
//                    output.append(","+avg+","+stdev);
//                }else{
//                    // append a dash or something
//                    output.append(",-,-");
//                }
//
//            }
//            output.append("\n");
//        }
//
//        FileWriter outFile = new FileWriter("test.csv");
//        outFile.append(output);
//        outFile.close();
//    }
//
//    public static void parseClusterEETestResults() throws Exception{
//
//        StringBuilder output = new StringBuilder();
//        output.append("Dataset,");
//        for(int i = 1; i <=100; i++){
//            output.append(i+",");
//        }
//        output.append("avg,stdev\n");
//
//        double[] accs;
//        File result;
//        Scanner scan;
//        boolean writeResult;
//        double sum;
//        double avg;
//        double stdev;
//
//        StringBuilder datasetOutString;
//
//        for(String dataset:DataSets.fileNames){
////        String[] datasets = {"ItalyPowerDemand"};
////        for(String dataset:datasets){
//            output.append(dataset);
////            for(ElasticEnsembleCluster.ClassifierVariants classifier:classifiers){
//            writeResult = true;
//            accs = new double[100];
//            sum = 0;
//            for(int resampleId = 1; resampleId <=100; resampleId++){
//                result = new File("/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/05_classification_loocvDatasets_fullPropEE/eeClusterOutput_fullEE/"+dataset+"/"+dataset+"_"+resampleId+".txt");
////                    result = new File(clusterDir+dataset+"/"+dataset+"_"+f+"/"+dataset+"_"+f+"_"+classifier+".txt");
//                if(!result.exists()){
//                    writeResult=false;
//                    accs[resampleId-1]=-1;
//                    continue;
//                }
//                    // file has accuracy (0-1) on the first row, every subsequent row is prediction/actual[prob_c1, prob_c2,...,prob_cn]
//                    // we don't care about individual predictions at the minute, so just store the double in the first row
////                    scan = new Scanner()
//                scan = new Scanner(result);
//                scan.useDelimiter("\n");
//                accs[resampleId-1] = Double.parseDouble(scan.next().trim());
//                sum+=accs[resampleId-1];
//                scan.close();
//            }
//
//            if(writeResult){
//                avg = sum/100;
//                stdev = 0;
//                for(int i = 0; i < accs.length; i++){
//                    output.append(","+accs[i]);
//                    stdev += (accs[i]-avg)*(accs[i]-avg);
//                }
//                stdev/=100;
//                stdev = Math.sqrt(stdev);
//                output.append(","+avg+","+stdev);
//            }else{
//                // append a dash or something
//                for(int i = 0; i < accs.length; i++){
//                    output.append(","+accs[i]);
//                }
//                output.append(",-,-");
//            }
//
//
//            output.append("\n");
//        }
//
//        FileWriter outFile = new FileWriter("EEResultsByFold.csv");
//        outFile.append(output);
//        outFile.close();
//    }
//
//    public static void clusterTestClassification(String dataName, ClassifierVariants classifier, int resampleId) throws Exception{
//        new File("eeClusterOutput_testResults/"+dataName+"/"+dataName+"_"+resampleId+"/").mkdirs();
//
//        // check if test results already exist, skip if it does - TODO
//        if(new File("/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/03_classification_loocvDatasets_testResults/eeClusterOutput_testResults/"+dataName+"/"+dataName+"_"+resampleId+"/"+dataName+"_"+resampleId+"_"+classifier+".txt").exists()){
//            throw new Exception("Test results already exist for "+dataName+"_"+resampleId+" for "+classifier);
//        }
//
//
//
//
//        // check if training results exist, skip if not
//        if(!new File("/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/02_classification_loocvDatasets/eeClusterOutput/"+dataName+"/"+dataName+"_"+resampleId+"/"+dataName+"_"+resampleId+"_parsedOutput/"+dataName+"_"+resampleId+"_"+classifier+".txt").exists()){
//            throw new Exception("Relevant training data doesn't exist for "+dataName+"_"+resampleId+" for "+classifier);
//        }
//
//
//        // get train and test data, resampled by seed
//        String dataDir = "/gpfs/home/sjx07ngu/TSCProblems/";
//        Instances originalTrain = ClassifierTools.loadData(dataDir+dataName+"/"+dataName+"_TRAIN");
//        Instances originalTest = ClassifierTools.loadData(dataDir+dataName+"/"+dataName+"_TEST");
//
//        Instances[] trainAndTest = InstanceTools.resampleTrainAndTestInstances(originalTrain, originalTest, resampleId);
//
//        Instances train = trainAndTest[0];
//        Instances test = trainAndTest[1];
//
//        // transform the data (if necessary). Do this here so we only have to do it once, then copy for other folds
//        if(classifier.equals(ClassifierVariants.DDTW_R1_1NN)||classifier.equals(ClassifierVariants.DDTW_Rn_1NN)||classifier.equals(ClassifierVariants.WDDTW_1NN)){
//             DerivativeFilter d = new DerivativeFilter();
//             train = d.process(train);
//             test = d.process(test);
//        }
//
//        ElasticEnsembleCluster ee = new ElasticEnsembleCluster(dataName, resampleId);
//        ee.setPathToTrainingResults("/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/02_classification_loocvDatasets/");
//        // set only the relevant classifier, read in data using in-built functionality
//        ee.removeAllClassifiersFromEnsemble();
//        ee.addClassifierToEnsemble(classifier);
//        ee.buildClassifier(train);
//
//        // extract the classifier that we are interested in - because we want to access the distributionForInstance
//        kNN knn = ee.getBuiltClassifier(classifier);
//
//        // test classification
//        int correct = 0;
//        double[] distribution;
//        double bsfDist;
//        double bsfDistId;
//
//        StringBuilder lineBuilder;
//        StringBuilder output = new StringBuilder();
//
//        for(int i = 0; i < test.numInstances(); i++){
//            distribution = knn.distributionForInstance(test.instance(i));
//            bsfDist = -1;
//            bsfDistId = -1;
//            lineBuilder = new StringBuilder();
//            for(int d = 0; d < distribution.length; d++){
//                lineBuilder.append(distribution[d]+",");
//                if(distribution[d] > bsfDist){
//                    bsfDist = distribution[d];
//                    bsfDistId = d;
//                }
//            }
//            output.append(bsfDistId+"/"+test.instance(i).classValue()+"["+lineBuilder.toString().substring(0, lineBuilder.length()-1)+"]\n");
//            if(bsfDistId==test.instance(i).classValue()){
//                correct++;
//            }
//        }
//
//
//        FileWriter out = new FileWriter("eeClusterOutput_testResults/"+dataName+"/"+dataName+"_"+resampleId+"/"+dataName+"_"+resampleId+"_"+classifier+".txt");
//        out.append(((double)correct/test.numInstances())+"\n"+output);
//        out.close();
//
//    }
//
//    public static void clusterEETestClassification(String dataName, int resampleId) throws Exception{
//
//        // note:  this existing check isn't implemented on the cluster version that's running the first iteration of this code (written after submission and wasn't necessary to stop execution since no results can exist yet! (Except ItalyPowerDemand))
//        // point: need to check that it works ok after it's included; should be fine, but only the method without this clause has been tested proir to distribution
//
//        // first, check whether the results already exist, and if they do, that they actually contain results!
//        File existing = new File("eeClusterOutput_fullEE/"+dataName+"/"+dataName+"_"+resampleId+".txt");
//        if(existing.exists()){
//            if(existing.length()>0){
//                return;
//            }
//
//        }
//
//        // this method uses all classifiers, we can worry about resticting membership etc later
//        for(ClassifierVariants cv:ClassifierVariants.values()){
//
//
//
//        }
//
//
//        // build classifier from cv files- we're mostly interested in the weights as we're working proportionatly
//        ElasticEnsembleCluster ee = new ElasticEnsembleCluster(dataName, resampleId);
//        ee.setPathToTrainingResults("/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/02_classification_loocvDatasets/");
//        ee.turnAllClassifiersOn();
//
//        Instances train = ClassifierTools.loadData("/gpfs/home/sjx07ngu/TSCProblems/"+dataName+"/"+dataName+"_TRAIN");
//        Instances test = ClassifierTools.loadData("/gpfs/home/sjx07ngu/TSCProblems/"+dataName+"/"+dataName+"_TEST");
//
//        // get the correct resample
//        Instances[] resampled = InstanceTools.resampleTrainAndTestInstances(train, test, resampleId);
//        train = resampled[0];
//        test = resampled[1];
//
//        ee.buildClassifier(train);
//
//        // get weight vector out of the classifier (for proportional vote scheme)
//        double[] propWeights = ee.cvAccs;
//
//        // load in the test predictions for each of the classifiers, so we can then make a weighted vote
//        double[][] individualPredictions = new double[ee.finalClassifierTypes.length][test.numInstances()];
//
//        File testFile;
//        Scanner scan;
//        String[] lineParts;
//        for(int c = 0; c < ee.finalClassifierTypes.length;c++){
//
//            // load test file
//            testFile = new File("/gpfs/home/sjx07ngu/ElasticEnsembleClusterDevelopment/02_EEReboot/03_classification_loocvDatasets_testResults/eeClusterOutput_testResults/"+dataName+"/"+dataName+"_"+resampleId+"/"+dataName+"_"+resampleId+"_"+ee.finalClassifierTypes[c]+".txt");
//            scan = new Scanner(testFile);
//            scan.useDelimiter("\n");
//            scan.next(); // ignore the test accuracy
//            for(int i =0; i < test.numInstances(); i++){
//                lineParts = scan.next().split("\\[");
//                lineParts = lineParts[0].split("/");
//                if(Double.parseDouble(lineParts[1].trim())!=test.instance(i).classValue()){
//                    throw new Exception("Error: class mismatch in "+dataName+"_"+resampleId+" at instance "+(i+1)+" "+lineParts[1].trim()+" ---"+test.instance(i));
//                }
//                individualPredictions[c][i]=Double.parseDouble(lineParts[0]);
//            }
//            scan.close();
//        }
//
//        // test predictions should be loaded. Now for each instance, vote using the predictions and the cv weighting
//        double[] votes;
//        int numClasses = train.numClasses();
//
//        double bsfVote;
//        ArrayList<Integer> votedClasses;
//        double finalVote;
//
//        int correct =0;
//        StringBuilder output = new StringBuilder();
//        StringBuilder voteString;
//
//        for(int i = 0; i < test.numInstances(); i++){
//            votes = new double[numClasses];
//            // get weighted vote from each constituent
//            for(int c = 0; c < ee.finalClassifierTypes.length; c++){
//
//                // votes stores the vote for class 0, class 1, ..., class C.
//                // so get the prediction of classifier c on instance i, update that cell with a vote to the value of the classifier's weighting
//                votes[(int)individualPredictions[c][i]] += propWeights[c]/100;
//
//            }
//
//            // find best vote, randomly split any ties
//            voteString = new StringBuilder();
//            votedClasses = new ArrayList<>();
//            bsfVote = -1;
//            for(int c = 0; c < votes.length; c++){
//                if(votes[c] > bsfVote){
//                    bsfVote = votes[c];
//                    votedClasses = new ArrayList<>();
//                    votedClasses.add(c);
//                }else if(votes[c]==bsfVote){
//                    votedClasses.add(c);
//                }
//                voteString.append(votes[c]+",");
//            }
//            // should have the biggest by now, so if there's only one then return that. Else, randomly return an element from the array
//            if(votedClasses.size()==1){
//                finalVote = votedClasses.get(0);
//            }else{
//                finalVote = votedClasses.get(new Random(1).nextInt(votedClasses.size()));
//            }
//            output.append(finalVote+"/"+test.instance(i).classValue()+"["+voteString.substring(0, voteString.length()-1)+"]\n");
//
//            if(finalVote==test.instance(i).classValue()){
//                correct++;
//            }
//        }
//        double acc = (double)correct/test.numInstances();
//        new File("eeClusterOutput_fullEE/"+dataName+"/").mkdirs();
//        FileWriter outWriter = new FileWriter("eeClusterOutput_fullEE/"+dataName+"/"+dataName+"_"+resampleId+".txt");
//        outWriter.append(acc+"\n");
//        for(int c = 0; c < ee.finalClassifierTypes.length; c++){
//            outWriter.append(ee.finalClassifierTypes[c]+",");
//        }
//        outWriter.append("\n");
//        for(int c = 0; c < ee.finalClassifierTypes.length; c++){
//            outWriter.append(ee.cvAccs[c]+",");
//        }
//        outWriter.append("\n"+output);
//        outWriter.close();
//    }
//
//    public static void writeScripts_testClassification_individualClassifiers() throws Exception{
//        new File("eeClusterScripts/").mkdirs();
//
//        // included as these definitely won't have run by the time I'm doing this. Can change later
//        HashSet<String> datasetsToAvoid = new HashSet<>();
//        datasetsToAvoid.add("ElectricDevices");
//        datasetsToAvoid.add("FordA");
//        datasetsToAvoid.add("FordB");
//        datasetsToAvoid.add("HandOutlines");
//        datasetsToAvoid.add("LargeKitchenAppliances");
//        datasetsToAvoid.add("NonInvasiveFatalECG_Thorax1");
//        datasetsToAvoid.add("NonInvasiveFatalECG_Thorax2");
//        datasetsToAvoid.add("RefrigerationDevices");
//        datasetsToAvoid.add("ScreenType");
//        datasetsToAvoid.add("ShapesAll");
//        datasetsToAvoid.add("SmallKitchenAppliances");
//        datasetsToAvoid.add("StarLightCurves");
//        datasetsToAvoid.add("UWaveGestureLibrary_X");
//        datasetsToAvoid.add("UWaveGestureLibrary_Y");
//        datasetsToAvoid.add("UWaveGestureLibrary_Z");
//        datasetsToAvoid.add("UWaveGestureLibraryAll");
//
//        ArrayList<ClassifierVariants> classifiersToUse = new ArrayList<>();
//        classifiersToUse.add(ClassifierVariants.Euclidean_1NN);
//        classifiersToUse.add(ClassifierVariants.DTW_R1_1NN);
//        classifiersToUse.add(ClassifierVariants.DTW_Rn_1NN);
//        classifiersToUse.add(ClassifierVariants.WDTW_1NN);
//        classifiersToUse.add(ClassifierVariants.DDTW_R1_1NN);
//        classifiersToUse.add(ClassifierVariants.DDTW_Rn_1NN);
//        classifiersToUse.add(ClassifierVariants.WDDTW_1NN);
//        classifiersToUse.add(ClassifierVariants.LCSS_1NN);
//        classifiersToUse.add(ClassifierVariants.ERP_1NN);
//        classifiersToUse.add(ClassifierVariants.MSM_1NN);
//        classifiersToUse.add(ClassifierVariants.TWE_1NN);
//
//        String part1 = "#!/bin/csh\n\n#BSUB -q short\n#BSUB -J test";
//        String part2 = "[1-100]\n#BSUB -oo output/test";
//        String part3 = "%I.out\n#BSUB -eo error/test";
//        String part4 = "%I.err\n#BSUB -R \"rusage[mem=2048]\"\n#BSUB -M 4000\n\nmodule add java/jdk/1.8.0_31\n\njava -jar -Xmx2048m TimeSeriesClassification.jar testClassification ";
//        String part5 = " $LSB_JOBINDEX \"/gpfs/home/sjx07ngu/TSCProblems/\" ";
//
//
//        FileWriter out;
//        String[] timeLineParts;
//        String jobString, jobName;
//
//
//        FileWriter instructionsOut = new FileWriter("instructions.txt");
//        Scanner scan;
//        for(String dataset:datasets){
//            if(datasetsToAvoid.contains(dataset)){
//                continue; // avoiding datasets with > 20 hour runtime for now
//            }
//
//            // load timing info
//
//            for(ClassifierVariants classifier: classifiersToUse){
//
//
//
//                new File("eeClusterScripts/"+dataset).mkdir();
//
//                jobName = dataset+"_"+classifier;
//                jobString = part1+jobName+part2+jobName+part3+jobName+part4+dataset+part5+"1 "+classifier;
//                out = new FileWriter("eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub");
//                out.append(jobString);
//                out.close();
//
//                // load in times - this was meant to be for training but it'll probablt help here too
//                scan = new Scanner(new File("../01_timingExperiments/timeEstimations/timeEstimations_"+jobName+".txt"));
//                scan.useDelimiter("\n");
//                timeLineParts = scan.next().split(",");
//
//                if(!timeLineParts[1].trim().equalsIgnoreCase("1")){
//                    throw new Exception("somethings gone wrong here - should only be prepping for LOOCV experiments");
//                }
//
//                instructionsOut.append(timeLineParts[0]+",bsub < eeClusterScripts/"+dataset+"/"+dataset+"_"+classifier+".bsub\n");
//            }
//        }
//
//        instructionsOut.close();
//
//
//
//
//
//    }
//
//
//    public static void reportOnFinishedClassifiersAndDatasets() throws Exception{
//
//        // look for parsed output on all resamples for all classifiers, create a simple list of the datasets that are fully done
//
//        // maybe in the form of: dataset | classifier1 | classifier 2 | ... | classifier m
//        //                        Adiac  |    Done     |  Not Done    |     |    Done       - or maybe 100 for all resamples done, 99, etc..
//        StringBuilder st = new StringBuilder();
//        ClassifierVariants[] classifiers = ClassifierVariants.values();
//        ClassifierVariants classifier;
//        String dataset;
//        int parsedCount;
//
//        String dataLoc;
//
//        st.append("Dataset");
//        for(int c = 0; c < classifiers.length;c++){
//            st.append(","+classifiers[c]);
//        }
//        st.append("\n");
//
//
//        // intentionally using indexes instead of for each to make sure I know the order of everthing!
//        for(int d = 0; d < datasets.length;d++){
//            dataset = datasets[d];
//            st.append(dataset);
//            for(int c = 0; c < classifiers.length; c++){
//                classifier = classifiers[c];
//                parsedCount = 0;
//                for(int resample = 1; resample <= 100; resample++){
//                    dataLoc = "eeClusterOutput/"+dataset+"/"+dataset+"_"+resample+"/"+dataset+"_"+resample+"_parsedOutput/"+dataset+"_"+resample+"_"+classifier+".txt";
//                    if(new File(dataLoc).exists()){
//                        parsedCount++;
//                    }
//                }
//                st.append(","+parsedCount);
//            }
//            st.append("\n");
//        }
//
//        FileWriter out = new FileWriter("finishedDatasetsReport.txt");
//        out.append(st);
//        out.close();
//    }
//
//}
//</editor-fold>
