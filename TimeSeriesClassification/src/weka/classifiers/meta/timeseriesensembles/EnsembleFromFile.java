/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.meta.timeseriesensembles;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import utilities.ClassifierTools;
import utilities.DebugPrinting;
import utilities.SaveCVAccuracy;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Given a results directory and classifiers to use (the results for which lie in that directory),
 * will ensemble those classifier's classifications to form a collective classification
 * 
 * Unless a specific name for the ensemble is provided, will attempt to construct a name
 * by concatenating the names of it's constituents
 * 
 * ensembleIdentifier is (as of writing) only used for saving the cv accuracy of the ensemble (for SaveCVAccuracy)
 * 
 * @author James Large
 */
public class EnsembleFromFile implements Classifier, DebugPrinting, SaveCVAccuracy {
    

    protected String[] classifierNames = null;
    
    //stores for data that is read in
    protected double[] individualCvAccs;
    protected double[][] individualCvPreds;
    protected double[][] individualTestPreds;
    protected double[][][] individualTestDists;
    
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
    
    /**
     * This will also set ensembleIdentifier to a default value (the concatenated classifiernames)
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
    
    private ModulePredictions loadResultsFile(File file) throws Exception {
        ArrayList<Double> alpreds = new ArrayList<>();
        ArrayList<ArrayList<Double>> aldists = new ArrayList<>();
        
        Scanner scan = new Scanner(file);
        scan.useDelimiter("\n");
        scan.next();
        scan.next();
        double acc = Double.parseDouble(scan.next().trim());
        
        String [] lineParts = null;
        while(scan.hasNext()){
            lineParts = scan.next().split(",");

            if (lineParts == null || lineParts.length < 2)
                continue;
            
            alpreds.add(Double.parseDouble(lineParts[1].trim()));
            
            if (lineParts.length > 3) {//dist for inst is present
                ArrayList<Double> dist = new ArrayList<>();
                for (int i = 3; i < 3+numClasses; ++i)  //act,pred,[empty],firstclassprob.... therefore 3 start
                    dist.add(Double.parseDouble(lineParts[i].trim()));
                aldists.add(dist);
            }
        }
        
        scan.close();
        
        double [] preds = new double[alpreds.size()];
        for (int i = 0; i < alpreds.size(); ++i)
            preds[i]= alpreds.get(i);
        
        double[][] distsForInsts = null;
        if (aldists.size() != 0) {
            distsForInsts = new double[aldists.size()][aldists.get(0).size()];
            for (int i = 0; i < aldists.size(); ++i)
                for (int j = 0; j < aldists.get(i).size(); ++j) 
                    distsForInsts[i][j] = aldists.get(i).get(j);
            
        }
        
        return new ModulePredictions(acc, preds, distsForInsts);
    }
    
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**EnsembleFromFile TRAIN**");
        
        if (!fileReadingParametersInitialised)
            throw new Exception("Parameters for results file reading have not been initialised");
        
        numTrainInsts = data.numInstances();
        numClasses = data.numClasses();
        numAttributes = data.numAttributes();
        
        int correct;  
        this.individualCvAccs = new double[this.classifierNames.length];
        this.individualCvPreds = new double[this.classifierNames.length][];
        
        this.individualTestPreds = new double[this.classifierNames.length][];
        this.individualTestDists = new double[this.classifierNames.length][][];

        //will look for all files and report all that are missing, instead of bailing on the first file not found
        //just helps debugging/running experiments a little 
        boolean anyFilesNotFound = false;
        String exceptionString = "";
        
        //for each module
        for(int c = 0; c < this.classifierNames.length; c++){
            boolean trainResultsLoaded = false;
            boolean testResultsLoaded = false; 
            
            //try and load in the train/test results for this module
            File moduleTrainResultsFile = findResultsFile(classifierNames[c], "train");
            if (moduleTrainResultsFile != null) { 
                printlnDebug(classifierNames[c] + " train loading...");

                ModulePredictions res = loadResultsFile(moduleTrainResultsFile);
                individualCvAccs[c] = res.acc;
                individualCvPreds[c] = res.preds;

                trainResultsLoaded = true;
            }

            File moduleTestResultsFile = findResultsFile(classifierNames[c], "test");
            if (moduleTestResultsFile != null) { 
                //of course these results not actually used at all during training, 
                //only loaded for future use when classifying with ensemble
                printlnDebug(classifierNames[c] + " test loading...");

                ModulePredictions res = loadResultsFile(moduleTestResultsFile);
                individualTestPreds[c] = res.preds; 
                individualTestDists[c] = res.distsForInsts; //dists not used atm, will be at later date

                if (numTestInsts != res.preds.length)
                    if (numTestInsts == -1) //first time here
                        numTestInsts = res.preds.length;
                    else
                        throw new Exception("Test results file sizes disagree, " + classifierNames[0] + " reports " + numTestInsts + ", " + classifierNames[c] + " reports " + res.preds.length);

                testResultsLoaded = true;
            }
            
            if (!trainResultsLoaded) {
                exceptionString += "\nTrain results files for '" + classifierNames[c] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. "
                        + "Directory given:\n" + individualResultsFilesDirectory;
                anyFilesNotFound = true;
            }
            if (!testResultsLoaded) {
                exceptionString += "\nTest results files for '" + classifierNames[c] + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. "
                        + "Directory given:\n" + individualResultsFilesDirectory;
                anyFilesNotFound = true;
            }
        }
        
        if (anyFilesNotFound) 
            throw new Exception(exceptionString);
    
        //got module trainpreds, time to combine to find overall ensemble trainpreds 
        this.ensembleCvPreds = new double[numTrainInsts];
        
        double actual, pred;
        double bsfWeight;
        correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        
        //for each train inst
        for(int i = 0; i < numTrainInsts; i++){
            actual = data.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[numClasses];

            //for each module
            for(int m = 0; m < classifierNames.length; m++){
                
                //this module makes a vote, with weight equal to its overall accuracy, for it's predicted class
                weightByClass[(int)individualCvPreds[m][i]]+=this.individualCvAccs[m];
                
                //update max class weighting so far
                //if two classes are tied for first, record both
                if(weightByClass[(int)individualCvPreds[m][i]] > bsfWeight){
                    bsfWeight = weightByClass[(int)individualCvPreds[m][i]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(individualCvPreds[m][i]);
                }else if(weightByClass[(int)individualCvPreds[m][i]] == bsfWeight){
                    bsfClassVals.add(individualCvPreds[m][i]);
                }
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
            preds[(int)pred] += this.individualCvAccs[c];
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
        return maxIndex(dist);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        return maxIndex(dist);
    }
    
    protected double maxIndex(double[] dist) {
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
        
        String individualResultsDirectory = "testResults/";
        
        EnsembleFromFile eff = new EnsembleFromFile();
        eff.setDebugPrinting(true);
        eff.setFileReadingParameters(individualResultsDirectory, classifierNames, dataset, 0);
        eff.buildClassifier(train);
        
        double correct = 0;
        for (int i = 0; i < test.numInstances(); ++i) {
            double pred = eff.classifyInstance(test.get(i));
            if (pred == test.get(i).classValue())
                correct++;
        }
        
        System.out.println("\n acc for " + eff.ensembleIdentifier +" =" + (correct/test.numInstances()));
    }
}
