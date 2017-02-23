package weka.classifiers.meta.timeseriesensembles;

import fileIO.OutFile;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import tsc_algorithms.cote.HiveCoteModule;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.DebugPrinting;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import utilities.SaveCVAccuracy;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.timeseriesensembles.weightings.*;
import weka.classifiers.meta.timeseriesensembles.voting.*;
import weka.filters.timeseries.SAX;

/**
 *
 * In it's current form, can either build ensemble and classify from results files 
 * of its members, (call setResultsFileLocationParameters(...)) else by default
 * will build/train the ensemble members normally. It will not do any file writing
 * (e.g trainFold# files of its members) aside from the overall ensemble train file as 
 * defined by/related to SaveCVAccuracy, if that has been set up (setCVPath(...))
 * 
 * 
 * @author James Large (james.large@uea.ac.uk) , Jason Lines (j.lines@uea.ac.uk)
 *      
 */

public class HESCA extends EnsembleFromFile implements HiveCoteModule, SaveCVAccuracy, DebugPrinting {
    
    protected ModuleWeightingScheme weightingScheme = new TrainAcc();
    protected ModuleVotingScheme votingScheme = new MajorityVote();
    protected EnsembleModule[] modules;
    
    protected boolean setSeed = false;
    protected int seed;
    
    protected Classifier[] classifiers; //todo classifiers essentially stored in the modules
    //can get rid of this reference at some point 
    
    
    protected String[] classifierNames;
    protected String[] classifierParameters;
    
    protected final SimpleBatchFilter transform;
    protected Instances train;
    
    //save cv accuracy
    protected boolean writeEnsembleTrainingFile = false;
    protected String outputEnsembleTrainingPathAndFile;
    
    //data generated during buildclassifier
    protected double[][] ensembleCvDists;
    protected double[] ensembleCvPreds;
    protected double ensembleCvAcc;
    
    //train/test data info
    protected int numTrainInsts;
    protected int numAttributes;
    protected int numClasses;
    protected int testInstCounter = 0;
    protected int numTestInsts = -1;
    protected Instance prevTestInstance = null;
        
    protected static class ErrorReport {
        private boolean anyErrors; 
        private String errorLog;
        
        public ErrorReport(String msgHeader) {
            errorLog = msgHeader;
        }
        
        public void log(String errorMsg) {
            anyErrors = true;
            errorLog += errorMsg;
        }
        
        public void throwIfErrors() throws Exception {
            if (anyErrors)
                throw new Exception(errorLog);
        }
        
        public boolean isEmpty() { return anyErrors; };
        public String getLog() { return errorLog; };
    }
    
    public HESCA() {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = null;
        this.setDefaultClassifiers();
    }
    
    public HESCA(SimpleBatchFilter transform) {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = transform;
        this.setDefaultClassifiers();
    }
    
    public HESCA(Classifier[] classifiers, String[] classifierNames) {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = null;
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        
        setDefaultClassifierParameters();
    }
    
    public HESCA(SimpleBatchFilter transform, Classifier[] classifiers, String[] classifierNames) {
        this.ensembleIdentifier = "HESCA";
        
        this.transform = transform;
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        
        setDefaultClassifierParameters();
    }
    
    public Classifier[] getClassifiers(){ return classifiers;}
    
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames) {
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        setDefaultClassifierParameters();
    }
    
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames, String[] parameters) {
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
        this.classifierParameters = parameters;
    }
    
    protected final void setDefaultClassifierParameters(){
        classifierParameters = new String[classifiers.length];
        for (String s : classifierParameters)
            s = "internalHESCA";
    }
    
    public final void setDefaultClassifiers(){
        this.classifiers = new Classifier[8];
        this.classifierNames = new String[8];
        
        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[0] = k;
        classifierNames[0] = "NN";
            
        classifiers[1] = new NaiveBayes();
        classifierNames[1] = "NB";
        
        classifiers[2] = new J48();
        classifierNames[2] = "C4.5";
        
        SMO svml = new SMO();
        svml.turnChecksOff();
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        svml.setKernel(kl);
        if(setSeed)
            svml.setRandomSeed(seed);
        classifiers[3] = svml;
        classifierNames[3] = "SVML";
        
        SMO svmq =new SMO();
//Assumes no missing, all real valued and a discrete class variable        
        svmq.turnChecksOff();
        PolyKernel kq = new PolyKernel();
        kq.setExponent(2);
        svmq.setKernel(kq);
        if(setSeed)
           svmq.setRandomSeed(seed);
        classifiers[4] =svmq;
        classifierNames[4] = "SVMQ";
        
        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        if(setSeed)
           r.setSeed(seed);            
        classifiers[5] = r;
        classifierNames[5] = "RandF";
            
            
        RotationForest rf=new RotationForest();
        rf.setNumIterations(50);
        if(setSeed)
           rf.setSeed(seed);
        classifiers[6] = rf;
        classifierNames[6] = "RotF";
        
        classifiers[7] = new BayesNet();
        classifierNames[7] = "bayesNet";   
        
        
        setDefaultClassifierParameters();
    }
    
    public void setRandSeed(int seed){
        this.setSeed = true;
        this.seed = seed;
    }

//Constants that determine the number of CV folds
//    The number of folds is all pretty arbitrary, but practically makes little difference
//especially if RandomForest and RotationForest use OOB error    
    public static int MAX_NOS_FOLDS=100;
    public static int FOLDS1=20;
    public static int FOLDS2=10;
    public static int NUM_CASES_THRESHOLD1=300;
    public static int NUM_CASES_THRESHOLD2=200;
    public static int NUM_CASES_THRESHOLD3=100;
    public static int NUM_ATTS_THRESHOLD1=200;
    public static int NUM_ATTS_THRESHOLD2=500;

  
    public static int findNumFolds(Instances train){
        int numFolds = train.numInstances();
        if(train.numInstances()>=NUM_CASES_THRESHOLD1)
            numFolds=FOLDS2;
        else if(train.numInstances()>=NUM_CASES_THRESHOLD2 && train.numAttributes()>=NUM_ATTS_THRESHOLD1)
            numFolds=FOLDS2;
        else if(train.numAttributes()>=NUM_ATTS_THRESHOLD2)
            numFolds=FOLDS2;
        else if (train.numInstances()>=NUM_CASES_THRESHOLD3) 
            numFolds=FOLDS1;
        return numFolds;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**HESCA TRAIN**");
        
        //transform data if specified
        if(this.transform==null){
            this.train = data;
        }else{
            this.train = transform.process(data);
        }
        
        //init
        this.numTrainInsts = train.numInstances();
        this.numClasses = train.numClasses();
        this.numAttributes = train.numAttributes();
        
        //set up modules
        initialiseModules();
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);
        
        //combine modules to find overall ensemble trainpreds 
        performEnsembleCV(data);
        
        //[SaveCVAccuracy] if writing results of this ensemble (to be read later as an individual module of a meta ensemble, 
        //i.e cote or maybe a meta-hesca), write the full ensemble trainFold# file
        if(this.writeEnsembleTrainingFile)
            writeEnsembleCVResults(data);
        
        this.testInstCounter = 0; //prep for start of testing
    }
    
    protected void writeEnsembleCVResults(Instances data) throws IOException {
        StringBuilder output = new StringBuilder();

        output.append(data.relationName()).append(",").append(ensembleIdentifier).append(",train\n");
        output.append(this.getParameters()).append("\n");
        output.append(ensembleCvAcc).append("\n");

        for(int i = 0; i < numTrainInsts; i++){
            //realclassval,predclassval,[empty],probclass1,probclass2,...
            output.append(data.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append(",");
            for (int j = 0; j < numClasses; j++) 
                output.append(",").append(ensembleCvDists[i][j]);
            
            output.append("\n");
        }

        new File(this.outputEnsembleTrainingPathAndFile).getParentFile().mkdirs();
        FileWriter fullTrain = new FileWriter(this.outputEnsembleTrainingPathAndFile);
        fullTrain.append(output);
        fullTrain.close();
    }

    protected void initialiseModules() throws Exception {
        this.modules = new EnsembleModule[classifierNames.length];
        for (int m = 0; m < modules.length; m++)
            modules[m] = new EnsembleModule(classifierNames[m], classifiers[m], classifierParameters[m]);
        
        //currently will only have file reading ON or OFF (not load some files, train the rest) 
        //having that creates many, many, many annoying issues, especially when classifying test cases
        if (readIndividualsResults) {
            if (!resultsFilesParametersInitialised)
                throw new Exception("Trying to load HESCA modules from file, but parameters for results file reading have not been initialised");
            loadModules(); //will throw exception if a module cannot be loaded (rather than e.g training that individual instead)
        }
            
        else 
            trainModules();
    }
    
    protected void trainModules() throws Exception {
        //still contains some code from when we could some some modules/train the rest
        //left in case we go back to it. remaining code has essentially 0 overhead
        
        //prep cv 
        int numFolds = findNumFolds(train);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(numFolds);
        cv.buildFolds(train);

        for (EnsembleModule module : modules) {
            if (module.trainResults == null) {
                //train file for this classifier not found
                printlnDebug(module.getModuleName() + " performing cv...");
                module.trainResults = cv.crossValidateWithStats(module.getClassifier(), train);

                if (writeIndividualsResults) { //if we're doing tainFold# file writing
                    writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train"); //write results out
                    printlnDebug(module.getModuleName() + " writing train file...");
                }
            }
            if (module.testResults == null) {
                //test file for this clasifier not found
                //build it so we can classify test samples later
                printlnDebug(module.getModuleName() + " building...");
                module.getClassifier().buildClassifier(train);
            }
        }
    }
    
    protected void loadModules() throws Exception {
        //will look for all files and report all that are missing, instead of bailing on the first file not found
        //just helps debugging/running experiments a little 
        ErrorReport errors = new ErrorReport("Errors while loading modules from file. Directory given: " + individualResultsFilesDirectory);
        
        //for each module
        for(int m = 0; m < this.modules.length; m++){
            boolean trainResultsLoaded = false;
            boolean testResultsLoaded = false; 
            
            //try and load in the train/test results for this module
            File moduleTrainResultsFile = findResultsFile(classifierNames[m], "train");
            if (moduleTrainResultsFile != null) { 
                printlnDebug(classifierNames[m] + " train loading... " + moduleTrainResultsFile.getAbsolutePath());
                
                modules[m].trainResults = loadResultsFile(moduleTrainResultsFile, numClasses);

                trainResultsLoaded = true;
            }

            File moduleTestResultsFile = findResultsFile(classifierNames[m], "test");
            if (moduleTestResultsFile != null) { 
                //of course these results not actually used at all during training, 
                //only loaded for future use when classifying with ensemble
                printlnDebug(classifierNames[m] + " test loading..." + moduleTestResultsFile.getAbsolutePath());

                modules[m].testResults = loadResultsFile(moduleTestResultsFile, numClasses);

                numTestInsts = modules[m].testResults.predClassVals.length;
                testResultsLoaded = true;
            }
            
            if (!trainResultsLoaded) {
                errors.log("\nTRAIN results files for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            }
            else if (votingScheme.getRequiresDistsForInstances() && modules[m].trainResults.distsForInsts == null) {
                errors.log("\nNo distribution for instance data found in TRAIN results file for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");
            }
        
            if (!testResultsLoaded) {
                errors.log("\nTEST results files for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            }
            else if (votingScheme.getRequiresDistsForInstances() && modules[m].trainResults.distsForInsts == null) {
                errors.log("\nNo distribution for instance data found in TEST results file for '" + classifierNames[m] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");
            }
        }
        
        errors.throwIfErrors();
    }
    
    protected void performEnsembleCV(Instances data) throws Exception {
        this.ensembleCvPreds = new double[numTrainInsts];
        this.ensembleCvDists = new double[numTrainInsts][];
        
        double actual, pred = .0;
        double bsfWeight;
        int correct = 0;
        ArrayList<Integer> bsfClassVals;
        double[] weightByClass;
        
        //for each train inst
        for(int i = 0; i < numTrainInsts; i++){
            actual = data.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -(Double.MAX_VALUE);
            
            //normalises so all sum to one internally, TODO maybe consider this further, 
            //any cases where we dont want to normalise yet?
            //will have to before setting the ensembleCVdist, but 
            //maybe something we want to do in the meantime
            weightByClass = votingScheme.distributionForTrainInstance(modules, i); 
            
            for (int c = 0; c < weightByClass.length; c++) {
                if(weightByClass[c] > bsfWeight){
                    bsfWeight = weightByClass[c];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(c);
                }else if(weightByClass[c] == bsfWeight){
                    bsfClassVals.add(c);
                }
            }
            
            
            if(bsfClassVals == null) {
                throw new Exception("bsfClassVals == null, NaN problem");
//                pred = new Random().nextInt(numClasses);
            }
            
            //if there's a tie for highest voted class after all module have voted, settle randomly
            if(bsfClassVals.size()>1)
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            else
                pred = bsfClassVals.get(0);
    
            //and make ensemble prediction
            if(pred==actual)
                correct++;
            
            this.ensembleCvPreds[i] = pred;
            this.ensembleCvDists[i] = weightByClass;
        }
        this.ensembleCvAcc = (double)correct/numTrainInsts;
    }
    
    /**
     * If building individuals from scratch, i.e not read results from files, call this
     * after testing is complete to build each module's testResults (accessible by module.testResults)
     * 
     * This will be done internally anyway if writeIndividualTestFiles(...) is called, this method
     * is made public only so that results can be accessed from memory during the same run if wanted 
     */
    public void finaliseIndividualModuleTestResults(double[] testSetClassVals) throws Exception {
        for (EnsembleModule module : modules)
            module.testResults.finaliseTestResults(testSetClassVals); //converts arraylists to double[]s and preps for writing
    }
    
    /**
     * @param throwExceptionOnFileParamsNotSetProperly added to make experimental code smoother, 
     *  i.e if false, can leave the call to writeIndividualTestFiles(...) in even if building from file, and this 
     *  function will just do nothing. else if actually intending to write test results files, pass true
     *  for exceptions to be thrown in case of genuine missing parameter settings
     * @throws Exception 
     */
    public void writeIndividualTestFiles(double[] testSetClassVals, boolean throwExceptionOnFileParamsNotSetProperly) throws Exception {
        if (!writeIndividualsResults || !resultsFilesParametersInitialised) {
            if (throwExceptionOnFileParamsNotSetProperly)
                throw new Exception("to call writeIndividualTestFiles(), must have called setResultsFileLocationParameters(...) and setWriteIndividualsResultsFiles()");
            else 
                return; //do nothing
        }
        
        finaliseIndividualModuleTestResults(testSetClassVals);
        
        for (EnsembleModule module : modules)
            writeResultsFile(module.getModuleName(), module.getParameters(), module.testResults, "test");
    }

    public String[] getClassifierNames() {
        return classifierNames;
    }

    public double[] getEnsembleCvPreds() {
        return ensembleCvPreds;
    }

    public double getEnsembleCvAcc() {
        return ensembleCvAcc;
    }

    public String getEnsembleIdentifier() {
        return ensembleIdentifier;
    }
    
    public void setEnsembleIdentifier(String ensembleIdentifier) {
        this.ensembleIdentifier = ensembleIdentifier;
    }

    public double[][] getPosteriorIndividualWeights() {
        double[][] weights = new double[modules.length][];
        for (int m = 0; m < modules.length; ++m) 
            weights[m] = modules[m].posteriorWeights;
        
        return weights;
    }

    public ModuleVotingScheme getVotingScheme() {
        return votingScheme;
    }

    public void setVotingScheme(ModuleVotingScheme votingScheme) {
        this.votingScheme = votingScheme;
    }
    
    public ModuleWeightingScheme getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(ModuleWeightingScheme weightingScheme) {
        this.weightingScheme = weightingScheme;
    }

    public double[] getIndividualCvAccs() {
        double [] accs = new double[modules.length];
        for (int i = 0; i < modules.length; i++) 
            accs[i] = modules[i].trainResults.acc;

        return accs;
    }
    
    //    public double[] getPriorIndividualWeights() {
//        return priorIndividualWeights;
//    }
//    
//    public void setPriorIndividualWeights(double[] priorWeights) {
//        this.priorIndividualWeights = priorWeights;
//    }
//            
//    private void setDefaultPriorWeights() {
//        priorIndividualWeights = new double[classifierNames.length];
//        for (int i = 0; i < priorIndividualWeights.length; i++)
//            priorIndividualWeights[i] = 1;
//    }
    

//    @Override
    public double[][] getIndividualCvPredictions() {
        double [][] preds = new double[modules.length][];
        for (int i = 0; i < modules.length; i++) 
            preds[i] = modules[i].trainResults.predClassVals;
        return preds;
    }
    
    public SimpleBatchFilter getTransform(){
        return this.transform;
    }
    
    @Override
    public void setCVPath(String pathAndName){
        this.outputEnsembleTrainingPathAndFile = pathAndName;
        this.writeEnsembleTrainingFile = true;
    }     
    
    @Override
    public String getParameters(){
        StringBuilder out = new StringBuilder();
        out.append("NA,");
        for(int m = 0; m < modules.length; m++){
            out.append(modules[m].getModuleName()).append("(").append(modules[m].priorWeight);
            for (int j = 0; j < modules[m].posteriorWeights.length; ++j)
                out.append("/").append(modules[m].posteriorWeights[j]);
            out.append("),");
        }
        return out.toString();
    }
    
//    public void readParameters(String paramLine) { 
//        String[] classifiers = paramLine.split(",");
//        
//        String[] classifierNames = new String[classifiers.length];
//        double[] priorWeights = new double[classifiers.length];
//        double[] postWeights = new double[classifiers.length];
//        
//        for (int i = 0; i < classifiers.length; ++i) {
//            String[] parts = classifiers[i].split("(");
//            classifierNames[i] = parts[0];
//            String[] weights = parts[1].split("/");
//            priorWeights[i] = Integer.parseInt(weights[0]);
//            for (int j = 1; j < weights.length; ++j)
//                postWeights[j-1] = Integer.parseInt(weights[j]);
//        }
//        
//    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = transform.process(rawContainer);
            ins = converted.instance(0);
        }
        
        if (testInstCounter == 0 && prevTestInstance == null) //definitely the first call, not e.g the first inst being classified for the second time
            printlnDebug("\n**TEST**");
        
        if (testInstCounter >= numTestInsts) //if no test files loaded, numTestInsts == -1
            throw new Exception("Received more test instances than expected, when loading test results files, found " + numTestInsts + " test cases");
                
        double[] preds;
        if (readIndividualsResults)//have results loaded from file
            preds = votingScheme.distributionForTestInstance(modules, testInstCounter);
        else //need to classify them normally
            preds = votingScheme.distributionForInstance(modules, ins);
        
        if (prevTestInstance != instance)
            ++testInstCounter;
        prevTestInstance = instance;
        
        return preds;
    }
    
//    /**
//     * Will try to use each individual's loaded test predictions (via the testInstIndex)
//     */
//    protected double[] distributionForInstance(int testInstIndex) throws Exception{      
//        return votingScheme.distributionForTestInstance(modules, testInstIndex);
//    }
    
//    /**
//     * Will try to use each individual's loaded test predictions (via the testInstIndex), else will find distribution normally (via testInst)
//     */
//    protected double classifyInstance(int testInstIndex) throws Exception{     
//        double[] dist = distributionForInstance(testInstIndex);
//        return indexOfMax(dist);
//    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        return indexOfMax(dist);
    }
    
    protected static double indexOfMax(double[] dist) {
        double max = dist[0];
        double maxInd = 0;
        
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        return maxInd;
    }
    
    /**
     * @return the predictions of each individual module, i.e [0] = first module's vote, [1] = second...
     */
    public double[] classifyInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = transform.process(rawContainer);
            ins = converted.instance(0);
        }
        
        double[] predsByClassifier = new double[this.classifiers.length];
                
        for(int i=0;i<classifiers.length;i++){
            predsByClassifier[i] = classifiers[i].classifyInstance(ins);
        }
        
        return predsByClassifier;
    }
    
    /**
     * @return the distributions of each individual module, i.e [0] = first module's dist, [1] = second...
     */
    public double[][] distributionForInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = transform.process(rawContainer);
            ins = converted.instance(0);
        }
        
        double[][] distsByClassifier = new double[this.classifiers.length][];
                
        for(int i=0;i<classifiers.length;i++){
            distsByClassifier[i] = classifiers[i].distributionForInstance(ins);
        }
        
        return distsByClassifier;
    }
    
    /**
     * classifiers/cnames are optional, leave null for default classifiers
     * 
     * todo: is a bodge job from previous code, before 'the big refactor'
     * 
     * clean it up at some point and use the up to date methods
     */
    public static void buildAndWriteFullIndividualTrainTestResults(Instances defaultTrainPartition, Instances defaultTestPartition, 
            String resultOutputDir, String datasetIdentifier, String ensembleIdentifier, int resampleIdentifier, 
            Classifier[] classifiers, String[] cNames,
            SimpleBatchFilter transform, boolean setSeed, boolean resample) throws Exception{
        HESCA h;
        if(classifiers != null) 
            h = new HESCA(transform, classifiers, cNames);
        else 
            h = new HESCA(transform);
        
        Instances train = new Instances(defaultTrainPartition);
        Instances test = new Instances(defaultTestPartition);
        if(resample && resampleIdentifier >0){
            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleIdentifier);
            train = temp[0];
            test = temp[1];
        }
        
        if (setSeed)
            h.setRandSeed(resampleIdentifier);
                
        h.setCVPath(resultOutputDir+h.ensembleIdentifier+"/Predictions/"+datasetIdentifier+"/trainFold"+resampleIdentifier+".csv");
        h.buildClassifier(train);
        
        StringBuilder[] byClassifier = new StringBuilder[h.classifiers.length+1];
        for(int c = 0; c < h.classifiers.length+1; c++){
            byClassifier[c] = new StringBuilder();
        }
        int[] correctByClassifier = new int[h.classifiers.length+1];
        
        double cvSum = 0;
        for(int c = 0; c < h.classifiers.length; c++){
            cvSum+= h.modules[c].trainResults.acc;
        }
        int correctByEnsemble = 0;
                
        double act;
        double pred;
        double[] preds;
        double[][] dists;
        
        
        double[] distForIns = null;
        double bsfClassVal = -1;
        double bsfClassWeight = -1;
        for(int i = 0; i < test.numInstances(); i++){
            act = test.instance(i).classValue();
            
            dists = h.distributionForInstanceByConstituents(test.instance(i));
            preds = new double[dists.length];
            
            for (int j = 0; j < dists.length; j++)
                preds[j] = indexOfMax(dists[j]);
                
            distForIns = new double[test.numClasses()];
            bsfClassVal = -1;
            bsfClassWeight = -1;
            for(int c = 0; c < h.classifiers.length; c++){
                byClassifier[c].append(act).append(",").append(preds[c]).append(",");
                for (int j = 0; j < test.numClasses(); j++)
                    byClassifier[c].append(",").append(dists[c][j]);
                byClassifier[c].append("\n");
                
                if(preds[c]==act){
                    correctByClassifier[c]++;
                }
                
                distForIns[(int)preds[c]]+= h.modules[c].trainResults.acc;
                if(distForIns[(int)preds[c]] > bsfClassWeight){
                    bsfClassVal = preds[c];
                    bsfClassWeight = distForIns[(int)preds[c]];
                }
            }
            
            if(bsfClassVal==act){
                correctByEnsemble++;
            }
            byClassifier[h.classifiers.length].append(act+","+bsfClassVal+",");
            for(int cVal = 0; cVal < distForIns.length; cVal++){
                byClassifier[h.classifiers.length].append(","+distForIns[cVal]/cvSum);
            }
            byClassifier[h.classifiers.length].append("\n");
        }
        
        h.setResultsFileLocationParameters(resultOutputDir, datasetIdentifier, resampleIdentifier); //dat hack... set after building/testing
        
        for (EnsembleModule module : h.modules)
            h.writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train");
        
        FileWriter out;
        String outPath;
        for(int c = 0; c < h.classifiers.length; c++){
            outPath = h.individualResultsFilesDirectory+h.classifierNames[c]+"/Predictions/"+h.datasetName;
            new File(outPath).mkdirs();
            out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
            out.append(h.datasetName+","+h.ensembleIdentifier+h.classifierNames[c]+",test\n");
            out.append("noParamInfo\n");
            out.append((double)correctByClassifier[c]/test.numInstances()+"\n");
            out.append(byClassifier[c]);
            out.close();            
        }
        
        outPath = h.individualResultsFilesDirectory+h.ensembleIdentifier+"/Predictions/"+h.datasetName;
        new File(outPath).mkdirs();
        out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
        out.append(h.datasetName+","+h.ensembleIdentifier+",test\n");
        out.append("noParamInfo\n");
        out.append((double)correctByEnsemble/test.numInstances()+"\n");
        out.append(byClassifier[h.classifiers.length]);
        out.close(); 
        
    }
    
    public static void exampleUseCase() throws Exception {
        String datasetName = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("c:/tsc problems/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/"+datasetName+"/"+datasetName+"_TEST");
        
        SimpleBatchFilter transform = new SAX();
        HESCA hesca = new HESCA(transform); //default classifiers, can still use different transforms if wanted
        
        
        Classifier[] classifiers = new Classifier[] { new kNN() };
        String [] names = new String[] { "NN" };
        String [] params = new String[] { "k=100" };
        
        hesca = new HESCA(classifiers, names); //to specify classsifiers, either this
        hesca.setClassifiers(classifiers, names); //or this 
        hesca.setClassifiers(classifiers, names, params); //can also pass parameter strings for each if wanted, defaults to "internalHESCA" otherwise
        //does not check for \n characters or anything (which would break file format), just dont pass them
        
        hesca.setDefaultClassifiers();//default ones still old/untuned, for old results recreation/correctness testing purposes
        
        //default uses train acc weighting and majority voting (as before) 
        hesca.setWeightingScheme(new TrainAccByClass()); //or set new methods
        hesca.setVotingScheme(new MajorityVote()); //some voting schemes require dist for inst to be defined
        
        //some old results files may not have written them, will break in this case
        //buildAndWriteFullIndividualTrainTestResults(...) will produce files with dists if needed (outside of a normal
        //hesca experiemental run)
        
        int resampleID = 0;
        //use this to set the location for any results file reading/writing
        hesca.setResultsFileLocationParameters("hescaTest/", datasetName, resampleID);
        
        //include this to turn on file reading, will read from location provided in setResultsFileLocationParameters(...)
        hesca.setWriteIndividualsResultsFiles(true);
        //include this to turn on file writing for individuals trainFold# files 
        hesca.setBuildIndividualsFromResultsFiles(true);
        //can only have one of these (or neither) set to true at any one time (internally, setting one to true 
        //will automatically set the other to false)
        //if building classifiers from scratch and wriitng to file, you obiovusly arnt reading results files
        //likewise if reading from file, to write would only replace the existing files with identical copies
        
        hesca.buildClassifier(train);
        
        //test however you would a normal classifier, no file writing here
        System.out.println(ClassifierTools.accuracy(test, hesca));
        
        //use this after testing is complete to write the testFold# files of the individuals (if setResultsFileLocationParameters(...) has been called)
        //see the java doc on this func for reasoning behind the boolean
        boolean throwExceptionOnFileParamsNotSetProperly = false;
        hesca.writeIndividualTestFiles(test.attributeToDoubleArray(test.classIndex()), throwExceptionOnFileParamsNotSetProperly);
    }
    
    public static void main(String[] args) throws Exception {
//        debugTest();
        schemeTests("C:/JamesLPHD/SmallUCRHESCAResultsFiles/");
//        exampleUseCase(); //look here for how to use, below is my random testing
        System.exit(0);
        
        String dataset = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");
        
        String[] classifierNames = new String[] {
            "bayesNet",
            "C4.5",
            "NB",
            "NN",
            "RandF",
            "RotF",
            "SVML",
            "SVMQ"
        };
        
        String individualResultsDirectory = "testResultsWithDists/";
        
//        buildAndWriteFullIndividualTrainTestResults(train, test, individualResultsDirectory, dataset, "", 0, null, true);
//        System.exit(0);
        
        HESCA h = new HESCA();
        h.setDebugPrinting(true);
        
//        h.setWeightingScheme(new EqualWeighting());
//        h.setWeightingScheme(new MCCWeighting());
//        h.setWeightingScheme(new FScore(0.5));
//        h.setWeightingScheme(new FScore(1));
//        h.setWeightingScheme(new FScore(2));
//        h.setWeightingScheme(new TrainAcc());
//        h.setWeightingScheme(new TrainAccByClass());
//        h.setWeightingScheme(new ConfusionEntropy()); 
//        h.setWeightingScheme(new ConfusionEntropyByClass()); 
        
//        h.setVotingScheme(new MajorityVote());
        h.setVotingScheme(new MajorityVoteByConfidence());
//        h.setVotingScheme(new MajorityConfidence());
        
        h.setResultsFileLocationParameters(individualResultsDirectory, dataset, 0);
        h.buildClassifier(train);
        
        System.out.println("weights:");
        for (int i = 0; i < classifierNames.length; i++)
            System.out.println(classifierNames[i] + ": " + Arrays.toString(h.getPosteriorIndividualWeights()[i]));
        
        double correct = 0;
        for (int i = 0; i < test.numInstances(); ++i) {
            double pred = h.classifyInstance(test.get(i));
            if (pred == test.get(i).classValue())
                correct++;
        }
        
        System.out.println("\nacc for " + h.ensembleIdentifier +" =" + (correct/test.numInstances()));
            
        
//        HESCA h1 = new HESCA();
//        h1.setRandSeed(0);
//        h1.buildClassifier(train);
//        
//        correct = 0;
//        for (int i = 0; i < test.numInstances(); ++i) {
//            double pred = h1.classifyInstance(test.get(i));
//            if (pred == test.get(i).classValue())
//                correct++;
//        }
//        
//        System.out.println("\nacc for " + h.ensembleIdentifier +" =" + (correct/test.numInstances()));
//        
//        for (int i = 0; i < train.numInstances(); ++i){
//            for (int j = 0; j < classifierNames.length; ++j){
//                if (h.modules[j].trainResults.predClassVals[i] != h1.modules[j].trainResults.predClassVals[i])
//                    System.out.println("difference in pred " + classifierNames[j] + " on " + i);
//            }
//        }
//        
//        for (int i = 0; i < classifierNames.length; i++) {
//            if (h.getPosteriorIndividualWeights()[i][0] != h1.getPosteriorIndividualWeights()[i][0])
//                System.out.println("difference in weighting " + classifierNames[i]);
//        }
//            
//        
//        System.out.println("");
//        Instances train = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
//        Instances test = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
//        
//        
//        buildAndWriteFullIndividualTrainTestResults(train, test, "testResults/", "ItalyPowerDemand", "", 0, null, true);
//        HESCA h = new HESCA();
//        h.setRandSeed(0);
//        h.setDebugPrinting(true);
//        h.turnOnResultsFileReadingWriting("testResults/", "", "ItalyPowerDemand", 0);
//        h.buildClassifier(train);
//        
//        double correct = 0;
//        for (int i = 0; i < test.numInstances(); ++i) {
//            double pred = h.classifyInstance(test.get(i), i);
//            if (pred == test.get(i).classValue())
//                correct++;
//        }
//        
//        System.out.println("\n acc=" + (correct/test.numInstances()));
//        
//        System.out.println("cv change test");
//        for (int a = 0; a < 10; ++a) {
//        
//            Instances train = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
//            Instances test = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
//
//            HESCA h = new HESCA();
//            h.setRandSeed(a);
//            HESCA_Local h1 = new HESCA_Local();
//            h1.setRandSeed(a);
//
//            h.buildClassifier(train);
//            System.out.println("build");
//            h1.buildClassifier(train);
//            System.out.println("build1");
//
//            //individualpreds
//            for (int i = 0; i < h.getIndividualCvPredictions().length; ++i) {
//                for (int j = 0; j < h.getIndividualCvPredictions()[i].length; ++j) {
//                    if (h.getIndividualCvPredictions()[i][j] != h1.getIndividualCvPredictions()[i][j]) {
//                        System.out.println("difference: " + i +" " + j +" " + h.getIndividualCvPredictions()[i][j] + " " + h1.getIndividualCvPredictions()[i][j]);
//                    }
//                }
//            }
//        }
        
        
        //old main
//        Instances train = ClassifierTools.loadData("c:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
//        Instances test = ClassifierTools.loadData("c:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
//        ShapeletTransform st = ShapeletTransformFactory.createTransform(train);
//        HESCA th = new HESCA(st);
////        EnhancedHESCA th = new EnhancedHESCA();
//        th.buildClassifier(train);
////        System.out.println(th.getEnsembleCvAcc());
////        double[] individualCvs = th.getIndividualCvAccs();
////        for(double acc:individualCvs){
////            System.out.print(acc+",");
////        }
//        
//        int correct = 0;
//        for(int i = 0; i < test.numInstances(); i++){
//            if(th.classifyInstance(test.instance(i))==test.instance(i).classValue()){
//                correct++;
//            }
//            System.out.println(th.classifyInstance(test.instance(i))+"\t"+test.instance(i).classValue());
//        }
//        System.out.println(correct+"/"+test.numInstances());
//        System.out.println((double)correct/test.numInstances());
    }       
    

    
    
    
    
    
    
    
    
    
    
    
    private static int numResamples = 25;
    
    private static final String[] smallishUCRdatasets  = { 
//        "Adiac",
//        "ArrowHead",
//        "CBF",
//        "ChlorineConcentration",
//        "Coffee",
//        "CricketX",
//        "CricketY",
//        "CricketZ",
//        "DiatomSizeReduction",
//        "DistalPhalanxOutlineAgeGroup",
//        "DistalPhalanxOutlineCorrect",
//        "DistalPhalanxTW",
        "ECG200",
//        "ItalyPowerDemand",
//        "MiddlePhalanxOutlineAgeGroup",
//        "MiddlePhalanxOutlineCorrect",
//        "MiddlePhalanxTW",
//        "MoteStrain",
//        "ProximalPhalanxOutlineAgeGroup",
//        "ProximalPhalanxOutlineCorrect",
//        "ProximalPhalanxTW",
//        "SonyAIBORobotSurface1",
//        "SonyAIBORobotSurface2",
//        "SyntheticControl",
//        "TwoLeadECG"
    };
    
    private static void buildBulkResultsFiles(String outpath) throws Exception {
        System.out.println("buildBulkResultsFiles()");
        
        double done = 0;
        for (String dset : smallishUCRdatasets) {
            Instances train = ClassifierTools.loadData("c:/tsc problems/"+dset+"/"+dset+"_TRAIN");
            Instances test = ClassifierTools.loadData("c:/tsc problems/"+dset+"/"+dset+"_TEST");
            
            for (int r = 0; r < numResamples; ++r) 
                HESCA.buildAndWriteFullIndividualTrainTestResults(train, test, outpath, dset, "", r, null, null, null, true, true); 
            
            System.out.println(((++done)/smallishUCRdatasets.length));
        }
    }
    
    
    private static void schemeTests(String individualResultsDirectory) throws Exception {
        System.out.println("schemeTests()");
        
        String[] classifierNames = new String[] {
            "bayesNet",
            "C4.5",
            "NB",
            "NN",
            "RandF",
            "RotF",
            "SVML",
            "SVMQ"
        };
        
        ModuleWeightingScheme[] weightSchemes = new ModuleWeightingScheme[] { 
//            new ConfusionEntropyByClass(), //broke
//            new ConfusionEntropy(),
//            new EqualWeighting(),
//            new MCCWeighting(),
            new TrainAcc(),
//            new TrainAccByClass(),
//            new FScore(),//1.0 default 
//            new FScore(0.5),
//            new FScore(2.0),
        };
        
        ModuleVotingScheme[] voteSchemes = new ModuleVotingScheme[] { 
//            new MajorityVote(),
            new MajorityVoteByConfidence(),
//            new MajorityConfidence(),
//            new NP_MAX(),
//            new AverageOfConfidences(),
//            new AverageVoteByConfidence(),
//            new ProductOfConfidences(),
//            new ProductOfVotesByConfidence()
        };
        
        
        double[][][] accs = new double[voteSchemes.length][weightSchemes.length][smallishUCRdatasets.length];
        
        for (int v = 0; v < voteSchemes.length; ++v) {
            ModuleVotingScheme vscheme = voteSchemes[v];
            
            for (int i = 0; i < weightSchemes.length; i++) {
                ModuleWeightingScheme wscheme = weightSchemes[i];

                for (int j = 0; j < smallishUCRdatasets.length; j++) {
                    String dataset = smallishUCRdatasets[j];

                    Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
                    Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");

                    double acc = .0;
                    for (int r = 0; r < numResamples; ++r)  {

                        Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, r);

                        HESCA h = new HESCA();
                        h.setDebugPrinting(true);
                        h.setWeightingScheme(wscheme);
                        h.setVotingScheme(vscheme);
                        h.getVotingScheme().setDebugPrinting(true);

                        h.setResultsFileLocationParameters(individualResultsDirectory, dataset, r);
                        h.setBuildIndividualsFromResultsFiles(true);
                        h.buildClassifier(data[0]);

                        double a = ClassifierTools.accuracy(data[1], h);
                        acc += a;
                    }

                    acc /= numResamples;
                    accs[v][i][j] = acc;
                }

                System.out.println("\t" + wscheme.getClass().getName() + " done");
            }
            
            System.out.println(vscheme.getClass().getName() + " done");
        }
        
        OutFile out = new OutFile("schemeTestsByVote2.csv");
        for (int v = 0; v < voteSchemes.length; v++) {
            out.writeLine("" + voteSchemes[v]);
        
            for (int w = 0; w < weightSchemes.length; w++) 
                out.writeString("," + weightSchemes[w]);
            out.writeLine("");

            for (int d = 0; d < smallishUCRdatasets.length; ++d) {
                out.writeString(smallishUCRdatasets[d]);

                for (int w = 0; w < weightSchemes.length; w++)
                    out.writeString("," + accs[v][w][d]);
                out.writeLine("");
            }
            
            out.writeLine("\n\n");
        }
        out.closeFile();
        
        OutFile out2 = new OutFile("schemeTestsByWeight2.csv");
         for (int w = 0; w < weightSchemes.length; w++) {
            out2.writeLine("" + weightSchemes[w]);
        
            for (int v = 0; v < voteSchemes.length; v++)
                out2.writeString("," + voteSchemes[v]);
            out2.writeLine("");

            for (int d = 0; d < smallishUCRdatasets.length; ++d) {
                out2.writeString(smallishUCRdatasets[d]);

                for (int v = 0; v < voteSchemes.length; v++)
                    out2.writeString("," + accs[v][w][d]);
                out2.writeLine("");
            }
            
            out2.writeLine("\n\n");
        }
        out2.closeFile();
        
        OutFile out3 = new OutFile("schemeTestsCombined2.csv");
        for (int v = 0; v < voteSchemes.length; v++)
            for (int w = 0; w < weightSchemes.length; w++) 
                out3.writeString("," + voteSchemes[v] + "+" + weightSchemes[w]);
        out3.writeLine("");

        for (int d = 0; d < smallishUCRdatasets.length; ++d) {
            out3.writeString(smallishUCRdatasets[d]);

            for (int v = 0; v < voteSchemes.length; v++)
                for (int w = 0; w < weightSchemes.length; w++)
                    out3.writeString("," + accs[v][w][d]);
            out3.writeLine("");
        }
        out3.closeFile();
    }
    
    public static void debugTest() throws Exception {
        
        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        Classifier[] cs = new Classifier[] { k };
        
        String[] cn = new String[] { "NN" };
        int fold = 5;
        String dset = "balloons";
        
        Instances all=ClassifierTools.loadData("C:/UCI Problems/"+dset+"/"+dset); 
        Instances[] data=InstanceTools.resampleInstances(all, fold, 0.5);
        
        buildAndWriteFullIndividualTrainTestResults(data[0], data[1], "lala/", dset, "asd", fold, cs, cn, null, true, true);
        
//        HESCA h = new HESCA(cs, cn);
//        h.setRandSeed(fold);
//        h.buildClassifier(data[0]);
    }
}
