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

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Scanner;

import utilities.ClassifierTools;

import tsc_algorithms.ElasticEnsemble;
import weka.classifiers.lazy.kNN;
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
    
    private String inputNameIdentifier;

    /**
     *
     * @param inputNameIdentifier
     */
    public ElasticEnsembleCluster(String inputNameIdentifier){
        super();
        this.inputNameIdentifier = inputNameIdentifier;
    }
    
    /**
     * An example use case of the cluster version for the Elastic Ensemble, run entirely on the local machine
     * 
     * @param tscDirPath The String path where the main TSC data repository is stored, e.g. "C:/Dropbox/TSC Problems"
     * @throws Exception
     */
    private static void exampleUseCase(String tscDirPath) throws Exception{
       
        Instances italyTrain = ClassifierTools.loadData(tscDirPath+"ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff");
        String resultsIdentifier = "italyTrainingExample";
        
        // 1) Generate individual prediction files for all classifiers with all possible parameter options 
        simulateClusterRun(italyTrain, resultsIdentifier);
        
        // 2) Parse individual outputs to find the best parameters for each classifier. Boolean option is whether to
        //    remove individual files once the best are written to file. If set to false, files will not be removed. 
        //    (Note: if the individual files are not removed at this stage, tidyUpIfParsedExists(String identifier)
        //    can be used to remove individual files later, but only if the parsed output for the classifier exists)
        writeAllBestParamFiles(resultsIdentifier,true);
        
        
        
        // 3) Use results to build a classifier
        ElasticEnsembleCluster eec = new ElasticEnsembleCluster(resultsIdentifier);
        eec.buildClassifier(italyTrain);
        
        // 4) Classifier is  equivilent to a locally-built ElasticEnsemble instance, use accordingly.
        Instances italyTest = ClassifierTools.loadData(tscDirPath+"ItalyPowerDemand/ItalyPowerDemand_TEST.arff");
        int correct = 0;
        for(int i = 0; i < italyTest.numInstances(); i++){
            if(eec.classifyInstance(italyTest.get(i))==italyTest.get(i).classValue()){
                correct++;
            }
        }
        System.out.println("Accuracy: " + new DecimalFormat("#.###").format((double)correct/italyTest.numInstances()*100)+"%");
 
    }
    
    /**
     * A method to simulate the experiments that would be run on Grace. Creates a dir in the project dir called eeClusterOutput
     * and generates a subdirectory for each measure type, each parameter option as a subdir of that, and then an individual txt
     * file for each instance.
     * 
     * @param instances Instances. The instances used for the cross-validation experiments
     * @param outputIdentifier String. A consistent identifier for all files relating to this set of experiments
     * @throws Exception
     */
    public static void simulateClusterRun(Instances instances, String outputIdentifier) throws Exception{
        
        ClassifierVariants[] classifiers = ClassifierVariants.values();
        
        int numJobs = 100*instances.numInstances(); 
        int instanceId, paramId;
               
        for(int c = 0; c < classifiers.length; c++){
            for(int j = 0; j < numJobs; j++){   // j is effectively the job number. Need numParamOptions and numInstances to split 
                paramId = j%100;
                instanceId = j/100;
                writeCvResultsForCluster(instances, outputIdentifier, classifiers[c], instanceId, paramId);
                if(!isParameterised(classifiers[c])){ // to avoid running non-parameterised measures multiple times
                    j+=99;
                }
            }
        }
    }
    
    private static boolean isParameterised(ClassifierVariants c){
        return !(c==ClassifierVariants.Euclidean_1NN || c == ClassifierVariants.DTW_R1_1NN || c== ClassifierVariants.DDTW_R1_1NN);
    }
    
    /**
     * The method that carries out a leave-one-out-cross-validation experiment for a single instance and writes the output to file.
     * @param instances Instances. The training instances
     * @param outputNameIdentifier String. A consistent name identifier for all jobs in this experiment
     * @param measureType ClassifierVariant. The elastic measure used with the nearest neighbour classifier in this experiment
     * @param instanceId int. The index of the test instance within the training data
     * @param paramId int. A reference to the job number for deriving which parameter options to use with this classifier
     * @throws Exception
     */
    public static void writeCvResultsForCluster(Instances instances, String outputNameIdentifier, ClassifierVariants measureType, int instanceId, int paramId) throws Exception{
        Instances train;
        if(measureType.equals(ClassifierVariants.DDTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_Rn_1NN)||measureType.equals(ClassifierVariants.WDDTW_1NN)){
            DerivativeFilter d = new DerivativeFilter();
            train = d.process(instances);
        }else{
            train = new Instances(instances);
        }
        Instance test = train.remove(instanceId);
        
        kNN classifier = getInternalClassifier(measureType, paramId, train);
        double prediction = classifier.classifyInstance(test);
        
        FileWriter out = null;
            
        File outDir = new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/");
        outDir.mkdirs();
        out = new FileWriter("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"/"+outputNameIdentifier+"_"+measureType+"_p_"+paramId+"_ins_"+instanceId+".txt");                
        out.append(prediction+"/"+test.classValue());
        out.close();        
        
        // create a global information file as a contingency (if it doesn't exist). This can provide information such as number of instances and actual class values
        // without needing to load raw data, and also help inturpret old files if the origin is unclear by storing the relation name
        String infoLoc = "eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+".info";
        if(!new File(infoLoc).exists()){
            out = new FileWriter(infoLoc);
            String summary = instances.toSummaryString();
            Scanner scan = new Scanner(summary);
            scan.useDelimiter("\n");
            out.append(scan.next().split(":")[1].trim()+"\n"); // relationName
            out.append(scan.next().split(":")[1].trim()+"\n"); // numInstances
            scan.close();
            for(int i = 0; i < instances.numInstances(); i++){
                out.append(instances.get(i).classValue()+"\n");
            }
            out.close();
        }
        
    }
    
    /**
     * Once all individual cross-validation experiments have been completer with the writeCvResultsForCluster method, this method parses the results to output files
     * that determine the best parameters for each classifier, within a set of results with the specofoed outputNameIdentifier
     * 
     * @param outputNameIdentifier String. A consistent reference for all individual results that form part of the same experiment.
     * @param measureType ClassifierVariant. The measure that results are to be summarised for.
     * @param tidyUp boolean. Determines whether to delete individual result files and directories once the best parameter is stored and written to a new output. 
     * @throws Exception
     */
    public static void writeBestParamFile(String outputNameIdentifier, ClassifierVariants measureType, boolean tidyUp) throws Exception{
        if(new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+measureType+".txt").exists()){
            System.err.println("Warning: Parsed output already exists for "+outputNameIdentifier+" and "+measureType+". Exiting method without writing to file.");
            return;
        }
        
        int expectedParams;
        if(measureType.equals(ClassifierVariants.Euclidean_1NN)||measureType.equals(ClassifierVariants.DTW_R1_1NN)||measureType.equals(ClassifierVariants.DDTW_R1_1NN)){
            expectedParams = 1;
        }else{
            expectedParams =100;
        }
        
        double[] paramPredictions;
        double[] bsfParamPredictions = null;
        
        int bsfParamId = -1;
        int correct;
        int bsfCorrect = -1;
        
        Scanner scan;
        String[] line;
        
        // get infoFile
        scan = new Scanner(new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+".info"));
        scan.useDelimiter("\n");
        scan.next(); // relationName
        int expectedInstances = Integer.parseInt(scan.next().trim());
        
        double[] classVals = new double[expectedInstances];
        for(int i = 0; i < expectedInstances; i++){
            classVals[i] = Double.parseDouble(scan.next().trim());
        }
        scan.close();
        
        for(int p = 0; p < expectedParams; p++){ // check accuracy of each parameter
            correct = 0;
            paramPredictions = new double[expectedInstances];
            for(int i = 0; i < expectedInstances; i++){
                // hardcode filename to be read in to avoid any index issues
                File results = new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType+"/"+outputNameIdentifier+"_"+measureType+"_p_"+p+"/"+outputNameIdentifier+"_"+measureType+"_p_"+p+"_ins_"+i+".txt");
                if(!results.exists()){
                    throw new Exception("Error: Missing results for "+measureType+" and paramId "+p+" on instance "+i);
                }
                scan = new Scanner(results);
                line = scan.next().trim().split("/");
                scan.close();
                paramPredictions[i] = Double.parseDouble(line[0]);
                if(paramPredictions[i]==classVals[i]){
                    correct++;
                }
                if(classVals[i]!=Double.parseDouble(line[1])){
                    throw new Exception("ERROR: class values have been confused. Instance "+i+ "should be "+classVals[i]+", file states "+Double.parseDouble(line[1]));
                }
            }
            if(correct>bsfCorrect){ // favours smaller params/earlioer options. This may cause different paramater selection to original approach for measures with two params - investigate further
                bsfCorrect = correct;
                bsfParamId = p;
                bsfParamPredictions = paramPredictions;                
            } 
        }
        
        new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_parsedOutput/").mkdirs();
        FileWriter out = new FileWriter("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+measureType+".txt");
        out.append((double)bsfCorrect/expectedInstances+"\n");
        out.append(bsfParamId+"\n");
        for(int i = 0; i < bsfParamPredictions.length; i++){
            out.append(bsfParamPredictions[i]+"/"+classVals[i]+"\n");
        }
        out.close();
        if(tidyUp){
            deleteDir(new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+measureType));
        }
    }
    
    private static void deleteDir(File dir){
        if(dir.isDirectory()){
            File[] files = dir.listFiles();
            for(int f = 0; f < files.length; f++){
                deleteDir(files[f]);
            }
        }
        dir.delete();
    }
    
    /**
     * 
     * A utility method for running the writeBestParamFile for all classifiers that are present in the outputNameIdentifier subdir of results
     * 
     * @param outputNameIdentifier String. A consistent reference for all individual results that form part of the same experiment.
     * @param tidyUp boolean. Determines whether to delete individual result files and directories once the best parameter is stored and written to a new output. 
     * @throws Exception
     */
    public static void writeAllBestParamFiles(String outputNameIdentifier, boolean tidyUp) throws Exception{
        File[] rawOutputsByClassifier = new File("eeClusterOutput/"+outputNameIdentifier).listFiles();
        for(int f = 0; f < rawOutputsByClassifier.length; f++){
            String fileName = rawOutputsByClassifier[f].getName();
            if(rawOutputsByClassifier[f].isDirectory() && !fileName.contains("parsedOutput")){
                writeBestParamFile(outputNameIdentifier, ClassifierVariants.valueOf(fileName.substring(outputNameIdentifier.length()+1)), tidyUp);
            }
        }
    }
    
    /**
     * A method to clean up individual output files if they remain after parsing. The work is logically identical to running writeBestParamFile with the tidyUp
     * option set to true. 
     * 
     * @param outputNameIdentifier String. A consistent reference for all individual results that form part of the same experiment.
     * @throws Exception
     */
    public static void tidyUpIfParsedExists(String outputNameIdentifier) throws Exception{
        
        ClassifierVariants[] classifiers = ClassifierVariants.values();
        for(int c = 0; c < classifiers.length; c++){
            if(new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_parsedOutput/"+outputNameIdentifier+"_"+classifiers[c]+".txt").exists()){
                // parsed output exists, so we can delete the relevant raw files
                deleteDir(new File("eeClusterOutput/"+outputNameIdentifier+"/"+outputNameIdentifier+"_"+classifiers[c]));
            }
        }            
    }
    
    
    private void loadCvFromFile(Instances train) throws Exception{

        // 1. check that results are available for all classifiers
        String[] line;
        
        for(int c = 0; c < this.finalClassifiers.length; c++){
            // should be in the form of:
            //      line 1: cv accuracy
            //      line 2: bestParamId - utility method included to convert this into the correct representation for a given classifier
            //      line 3 to m+2: prediction/actualClassValue (second part is redundant, but used as a validation measure)
            
            File resultFile = new File("eeClusterOutput/"+this.inputNameIdentifier+"/"+this.inputNameIdentifier+"_parsedOutput/"+this.inputNameIdentifier+"_"+finalClassifiers[c]+".txt");
            if(!resultFile.exists()){
                throw new Exception("Error: result file could not be loaded - "+"eeClusterOutput/"+this.inputNameIdentifier+"/"+this.inputNameIdentifier+"_parsedOutput/"+this.inputNameIdentifier+"_"+finalClassifiers[c]+".txt");
            }
            Scanner scan = new Scanner(resultFile);
            scan.useDelimiter("\n");
            this.cvAccs[c] = Double.parseDouble(scan.next().trim())*100; // original implementation is 0-100, not 0-1, so need to correct to match formatting
            this.bestParams[c] = getParamsFromParamId(this.finalClassifiers[c], Integer.parseInt(scan.next().trim()), train);
            for(int i = 0; i < train.numInstances(); i++){
                line = scan.next().split("/");
                this.cvPreds[c][i] = Double.parseDouble(line[0].trim());
                if(Double.parseDouble(line[1].trim())!=train.instance(i).classValue()){
                    throw new Exception("Error: class mismatch issue for instance "+(i+1)+". File: "+line[1]+", Instances: "+train.instance(i).classValue());
                }
            }
        }
        
        
    }
    
    
    @Override
    public void buildClassifier(Instances train) throws Exception {
        
        if(this.inputNameIdentifier == null){
            throw new Exception("Error: this classifier must be built using pre-exitsing crossvalidation results. Please set input identifier and ensure files are in the eeClusterResults dir of this project's files.");
        }
        
        this.finalClassifiers = new ClassifierVariants[classifiersToUse.size()];
        this.classifiersToUse.toArray(finalClassifiers);
        this.cvAccs = new double[this.finalClassifiers.length];
        this.cvPreds = new double[this.finalClassifiers.length][train.numInstances()];
        this.bestParams = new double[this.finalClassifiers.length][];
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
    
    public static void clusterMaster(String[] args) throws Exception{
        if(args[0].equalsIgnoreCase("start")){
//            clusterMaster(args[1], args[2]);
        }else if(args[0].equalsIgnoreCase("cvClassification")){
            
            String outputIdentifier = args[1];
            String instancesAddress = args[2];
            int paramId = Integer.parseInt(args[3].trim())-1;
            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[4]);
            
            Instances train = ClassifierTools.loadData(instancesAddress);
            for(int i = 0; i < train.numInstances(); i++){
                ElasticEnsembleCluster.writeCvResultsForCluster(train, outputIdentifier, classifier, i, paramId);
            }
        }else if(args[0].equalsIgnoreCase("parseResults")){
            
            String outputIdentifier = args[1];
            ElasticEnsembleCluster.ClassifierVariants classifier = ElasticEnsembleCluster.ClassifierVariants.valueOf(args[2]);
            ElasticEnsembleCluster.writeBestParamFile(outputIdentifier, classifier, true);
            
        }else{
            System.out.println("You shouldn't be here?!");
        }
    }
    
    /**
     * Main method
     * 
     * @param args String[] Arguments used to invoke the main logic of the program.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception{
        clusterMaster(args);
    }
    
}
