/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.meta.timeseriesensembles;

import fileIO.OutFile;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Consumer;
import utilities.ClassifierTools;
import utilities.DebugPrinting;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.meta.timeseriesensembles.weightings.*;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Given a results directory and classifiers to use (the results for which lie in that directory),
 * will ensemble those classifier's classifications to form a collective classification
 * 
 * By default, will use each member's train accuracy as it's vote weighting
 * 
 * Unless a specific name for the ensemble is provided, will attempt to construct a name
 * by concatenating the names of it's constituent. ensembleIdentifier is (as of writing) 
 * only used for saving the cv accuracy of the ensemble (for SaveCVAccuracy)
 * 
 * @author James Large
 */
public class EnsembleFromFile implements Classifier, DebugPrinting, SaveCVAccuracy {
   
    protected ModuleWeightingScheme weightingScheme = new TrainAccWeighting();
    
    protected String[] classifierNames = null;
    
    //stores for data that is read in
    protected double[] individualCvAccs;
    protected double[][] individualCvPreds;
    protected double[][] individualTestPreds;
    protected double[][][] individualTestDists;
    
    //each module makes a vote, with a weight defined for this classifier when predicting this class 
    //most weighting schemes will have weights for each class for a single classifier equal, but leaving open the
    //possibility of having e.g certain members being experts at classifying certains classes etc
    protected double[][] posteriorIndividualWeights; //[classifier][weightforclass]
    
    //by default (and i imagine in the vast majority of cases) all prior weights are equal (i.e 1)
    //however may be circumstances where certain classifiers are themselves part of 
    //a subensemble or something
    protected double[] priorIndividualWeights;

    //data generated during buildclassifier
    protected double[] ensembleCvPreds;
    protected double ensembleCvAcc;
    
    //train/test data info
    protected String datasetName;
    protected int numTrainInsts;
    protected int numAttributes;
    protected int numClasses;
    protected int testInstCounter = 0;
    protected int numTestInsts = -1;
    protected Instance prevTestInstance = null;

    //results file reading/writing
    protected boolean fileReadingParametersInitialised;
    protected String individualResultsFilesDirectory;
    protected String ensembleIdentifier;
    protected int resampleIdentifier;
    
    //savecvaccuracy
    protected boolean writeEnsembleTrainingFile = false;
    protected String outputEnsembleTrainingPathAndFile;

    public EnsembleFromFile() { 
        fileReadingParametersInitialised = false;
        
        this.individualResultsFilesDirectory = null;
        this.classifierNames = null;
        this.datasetName = null;
        this.resampleIdentifier = -1;
        
        this.priorIndividualWeights = null;
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
        return posteriorIndividualWeights;
    }
    

    
    
    public ModuleWeightingScheme getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(ModuleWeightingScheme weightingScheme) {
        this.weightingScheme = weightingScheme;
    }
    
    public double[] getPriorIndividualWeights() {
        return priorIndividualWeights;
    }
    
    public void setPriorIndividualWeights(double[] priorWeights) {
        this.priorIndividualWeights = priorWeights;
    }
            
    private void setDefaultPriorWeights() {
        priorIndividualWeights = new double[classifierNames.length];
        for (int i = 0; i < priorIndividualWeights.length; i++)
            priorIndividualWeights[i] = 1;
    }
    
    /**
     * This will also set ensembleIdentifier to a default value (the concatenated classifiernames with a & separator)
     * 
     * If a unique identifier is wanted, call setEnsembleIdentifier AFTER the call to this
     */
    public void setFileReadingParameters(String individualResultsFilesDirectory, String[] classifierNames, String datasetName, int resampleIdentifier) {
        fileReadingParametersInitialised = true;
        
        this.individualResultsFilesDirectory = individualResultsFilesDirectory;
        this.classifierNames = classifierNames;
        this.datasetName = datasetName;
        this.resampleIdentifier = resampleIdentifier;
        
        this.ensembleIdentifier = classifierNames[0];
        for (int i = 1; i < classifierNames.length; ++i)
            this.ensembleIdentifier += "&" + classifierNames[i];
        
        if (priorIndividualWeights == null)
            setDefaultPriorWeights();
    }
    
    @Override
    public void setCVPath(String pathAndName){
        this.outputEnsembleTrainingPathAndFile = pathAndName;
        this.writeEnsembleTrainingFile = true;
    }     
    
    @Override
    public String getParameters(){
        StringBuilder out = new StringBuilder();
        for(int c = 0; c < this.classifierNames.length; c++){
            out.append(classifierNames[c]+",");
        }
        return out.toString();
    }

    public File findResultsFile(String classifierName, String trainOrTest) {
        File file = new File(individualResultsFilesDirectory+classifierName+"/Predictions/"+datasetName+"/"+trainOrTest+"Fold"+resampleIdentifier+".csv");
        if(!file.exists() || file.length() == 0)
            return null;
        else 
            return file;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**EnsembleFromFile TRAIN**");
        
        if (!fileReadingParametersInitialised)
            throw new Exception("Parameters for results file reading have not been initialised");
        
        numTrainInsts = data.numInstances();
        numClasses = data.numClasses();
        numAttributes = data.numAttributes();
        
        this.individualCvAccs = new double[this.classifierNames.length];
        this.posteriorIndividualWeights = new double[this.classifierNames.length][];
        this.individualCvPreds = new double[this.classifierNames.length][];
        
        this.individualTestPreds = new double[this.classifierNames.length][];
        this.individualTestDists = new double[this.classifierNames.length][][];

        //will look for all files and report all that are missing, instead of bailing on the first file not found
        //just helps debugging/running experiments a little 
        boolean anyFilesNotFound = false;
        String exceptionString = "Directory given: " + individualResultsFilesDirectory;
        
        //for each module
        for(int c = 0; c < this.classifierNames.length; c++){
            boolean trainResultsLoaded = false;
            boolean testResultsLoaded = false; 
            
            //try and load in the train/test results for this module
            File moduleTrainResultsFile = findResultsFile(classifierNames[c], "train");
            if (moduleTrainResultsFile != null) { 
                printlnDebug(classifierNames[c] + " train loading... " + moduleTrainResultsFile.getAbsolutePath());

                ModulePredictions res = ModulePredictions.loadResultsFile(moduleTrainResultsFile, numClasses);
                individualCvAccs[c] = res.acc;
                individualCvPreds[c] = res.preds;
                
                posteriorIndividualWeights[c] = weightingScheme.defineWeighting(res, numClasses);
//                
//                for (double d : posteriorIndividualWeights[c])
//                    if (Double.compare(Double.NaN, d) == 0)
//                        throw new Exception("Weight is NaN, " + classifierNames[c] + ": " + Arrays.toString(posteriorIndividualWeights[c]));
//                
                trainResultsLoaded = true;
            }

            File moduleTestResultsFile = findResultsFile(classifierNames[c], "test");
            if (moduleTestResultsFile != null) { 
                //of course these results not actually used at all during training, 
                //only loaded for future use when classifying with ensemble
                printlnDebug(classifierNames[c] + " test loading..." + moduleTestResultsFile.getAbsolutePath());

                ModulePredictions res = ModulePredictions.loadResultsFile(moduleTestResultsFile, numClasses);
                //class vals found in results file obviously NOT used at all
                individualTestPreds[c] = res.preds; 
                individualTestDists[c] = res.distsForInsts; //dists not used atm, will be at later date

                numTestInsts = res.preds.length;
                testResultsLoaded = true;
            }
            
            if (!trainResultsLoaded) {
                exceptionString += "\nTrain results files for '" + classifierNames[c] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ";
                anyFilesNotFound = true;
            }
            if (!testResultsLoaded) {
                exceptionString += "\nTest results files for '" + classifierNames[c] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ";
                anyFilesNotFound = true;
            }
        }
        
        if (anyFilesNotFound) 
            throw new Exception(exceptionString);
    
        //got module trainpreds, time to combine to find overall ensemble trainpreds 
        this.ensembleCvPreds = new double[numTrainInsts];
        
        double actual, pred = .0;
        double bsfWeight;
        int correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        
        //for each train inst
        for(int i = 0; i < numTrainInsts; i++){
            actual = data.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -Double.MAX_VALUE;
            weightByClass = new double[numClasses];

            //for each module
            for(int m = 0; m < classifierNames.length; m++){
                
                //this module makes a vote, with a weight defined for this classifier when predicting this class 
                //most weighting schemes will have weights for each class for a single classifier equal, but leaving open the
                //possibility
                pred = individualCvPreds[m][i];
                weightByClass[(int)pred]+= priorIndividualWeights[m] * posteriorIndividualWeights[m][(int)pred];
//                weightByClass[(int)individualCvPreds[m][i]]+=this.individualCvAccs[m];
                
                //update max class weighting so far
                //if two classes are tied for first, record both
                
                if(weightByClass[(int)pred] > bsfWeight){
                    bsfWeight = weightByClass[(int)pred];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(pred);
                }else if(weightByClass[(int)pred] == bsfWeight){
                    bsfClassVals.add(pred);
                }
            }
            
            if (bsfClassVals == null) {
                System.out.println(numClasses);
                System.out.println(pred);
                System.out.println(weightByClass[(int)pred]);
                System.out.println(bsfWeight);
                System.out.println(Arrays.toString(weightByClass));
                System.out.println(bsfClassVals);
                throw new Exception("Problem with weight setting: all weights are 0, " + weightingScheme.getClass().getName());
            }
            
            //if there's a tie for highest voted class after all module have voted, settle randomly
            if(bsfClassVals.size()>1){
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            }else{
                pred = bsfClassVals.get(0);
            }
            
            //and make ensemble prediction
            if(pred==actual){
                correct++;
            }
            this.ensembleCvPreds[i]=pred;
        }
        this.ensembleCvAcc = (double)correct/numTrainInsts;
        
        //if writing results of this ensemble (to be read later as an individual module of a meta ensemble, 
        //i.e cote or maybe a meta-hesca), write the full ensemble trainFold# file
        if(this.writeEnsembleTrainingFile){
            StringBuilder output = new StringBuilder();

            output.append(data.relationName()).append(","+ensembleIdentifier+",train\n");
            output.append(this.getParameters()).append("\n");
            output.append(ensembleCvAcc).append("\n");

            for(int i = 0; i < numTrainInsts; i++){
                output.append(data.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append("\n");
            }

            new File(this.outputEnsembleTrainingPathAndFile).getParentFile().mkdirs();
            FileWriter fullTrain = new FileWriter(this.outputEnsembleTrainingPathAndFile);
            fullTrain.append(output);
            fullTrain.close();
        }
        
        testInstCounter = 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        if (testInstCounter == numTestInsts) //if no test files loaded, numTestInsts == -1
            throw new Exception("Received more test instances than expected, when loading test results files, found " + numTestInsts + " test cases");
                
        double[] preds = distributionForInstance(testInstCounter);
        
        if (prevTestInstance != instance)
            ++testInstCounter;
        prevTestInstance = instance;
        
        return preds;
    }
    
    /**
     * Will try to use each individual's loaded test predictions (via the testInstIndex)
     */
    public double[] distributionForInstance(int testInstIndex) throws Exception{      
        if (testInstIndex == 0 && prevTestInstance == null) //definitely the first call, not e.g the first inst being classified for the second time
            printlnDebug("\n**TEST**");
        
        double[] preds = new double[numClasses];
        
        double pred;
        for(int c = 0; c < classifierNames.length; c++){
            pred = individualTestPreds[c][testInstIndex]; 
//            preds[(int)pred] += this.individualCvAccs[c];
            preds[(int)pred] += priorIndividualWeights[c] * posteriorIndividualWeights[c][(int)pred];
        }
        
        //normalise so all sum to one 
        double sum=preds[0];
        for(int i=1;i<preds.length;i++)
            sum+=preds[i];
        for(int i=0;i<preds.length;i++)
            preds[i]/=sum;
        
        return preds;
    }
    
    /**
     * Will try to use each individual's loaded test predictions (via the testInstIndex), else will find distribution normally (via testInst)
     */
    public double classifyInstance(int testInstIndex) throws Exception{     
        double[] dist = distributionForInstance(testInstIndex);
        return indexOfMax(dist);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        return indexOfMax(dist);
    }
    
    protected double indexOfMax(double[] dist) {
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

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String [] args) throws Exception {   
//        buildBulkResultsFiles("E:/PHDEnsembleResFiles/BasicHesca/");
        weightingSchemeTests("E:/PHDEnsembleResFiles/BasicHesca/");
        
//        String dataset = "ItalyPowerDemand";
//        
//        Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
//        Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");
//        
//        String[] classifierNames = new String[] {
//            "bayesNet",
//            "C4.5",
//            "NB",
//            "NN",
//            "RandF",
//            "RotF",
//            "SVML",
//            "SVMQ"
//        };
//        
//        String individualResultsDirectory = "testResults/";
//        
//        EnsembleFromFile eff = new EnsembleFromFile();
//        eff.setDebugPrinting(true);
//        
////        eff.setWeightingScheme(new EqualWeighting());
////        eff.setWeightingScheme(new MCCWeighting());
////        eff.setWeightingScheme(new F1Weighting());
//        eff.setWeightingScheme(new TrainAccWeighting());
////        eff.setWeightingScheme(new CENWeighting()); 
////        eff.setWeightingScheme(new pCENAccWeighting()); //to implement
//        
//        eff.setFileReadingParameters(individualResultsDirectory, classifierNames, dataset, 0);
//        eff.buildClassifier(train);
//        
//        System.out.println("weights:");
//        for (int i = 0; i < classifierNames.length; i++)
//            System.out.println(classifierNames[i] + ": " + Arrays.toString(eff.getPosteriorIndividualWeights()[i]));
//        
//        double correct = 0;
//        for (int i = 0; i < test.numInstances(); ++i) {
//            double pred = eff.classifyInstance(test.get(i));
//            if (pred == test.get(i).classValue())
//                correct++;
//        }
//        
//        System.out.println("\nacc for " + eff.ensembleIdentifier +" =" + (correct/test.numInstances()));
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    public static int numResamples = 25;
    
    public static final String[] smallishUCRdatasets  = { //smallish
        "Adiac",
        "ArrowHead",
        "CBF",
        "ChlorineConcentration",
        "Coffee",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxOutlineCorrect",
        "DistalPhalanxTW",
        "ECG200",
        "ItalyPowerDemand",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxOutlineCorrect",
        "MiddlePhalanxTW",
        "MoteStrain",
        "ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxOutlineCorrect",
        "ProximalPhalanxTW",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "SyntheticControl",
        "TwoLeadECG"
    };
    
    public static void buildBulkResultsFiles(String outpath) throws Exception {
        System.out.println("buildBulkResultsFiles()");
        
        double done = 0;
        for (String dset : smallishUCRdatasets) {
            Instances train = ClassifierTools.loadData("c:/tsc problems/"+dset+"/"+dset+"_TRAIN");
            Instances test = ClassifierTools.loadData("c:/tsc problems/"+dset+"/"+dset+"_TEST");
            
            for (int r = 0; r < numResamples; ++r) 
                HESCA.buildAndWriteFullIndividualTrainTestResults(train, test, outpath, dset, "", r, null, true); 
            
            System.out.println(((++done)/smallishUCRdatasets.length));
        }
    }
    
    
    public static void weightingSchemeTests(String individualResultsDirectory) throws Exception {
        System.out.println("weightingSchemeTests()");
        
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
        
        ModuleWeightingScheme[] schemes = new ModuleWeightingScheme[] { 
            new cCENWeighting(),
            new CENWeighting(),
            new EqualWeighting(),
            new MCCWeighting(),
            new TrainAccWeighting(),
            new FScoreWeighting(),
            new FScoreWeighting(0.5),
            new FScoreWeighting(2.0),
        };
        
        double[][] accs = new double[schemes.length][smallishUCRdatasets.length];
        
        for (int i = 0; i < schemes.length; i++) {
            ModuleWeightingScheme scheme = schemes[i];
            
            for (int j = 0; j < smallishUCRdatasets.length; j++) {
                String dataset = smallishUCRdatasets[j];
                        
                Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
                Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");

                double acc = .0;
                for (int r = 0; r < numResamples; ++r)  {

                    Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, r);

                    EnsembleFromFile eff = new EnsembleFromFile();
//                    eff.setDebugPrinting(true);
                    eff.setWeightingScheme(scheme);

                    eff.setFileReadingParameters(individualResultsDirectory, classifierNames, dataset, r);
                    eff.buildClassifier(data[0]);

                    double a = ClassifierTools.accuracy(data[1], eff);
                    acc += a;
                    
//                    System.out.println(a);
                }
                
                acc /= numResamples;
                accs[i][j] = acc;
            }
            
            System.out.println(scheme.getClass().getName() + " done");
        }
        
        OutFile out = new OutFile("weightingschemeTests.csv");
        
        for (int i = 0; i < schemes.length; i++) 
            out.writeString("," + schemes[i]);
        out.writeLine("");
        
        for (int i = 0; i < smallishUCRdatasets.length; ++i) {
            out.writeString(smallishUCRdatasets[i]);
            
            for (int j = 0; j < schemes.length; j++)
                out.writeString("," + accs[j][i]);
            out.writeLine("");
        }
        
        out.closeFile();
    }
}
