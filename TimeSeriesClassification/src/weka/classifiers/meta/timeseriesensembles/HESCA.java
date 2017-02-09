/**
 * NOTE: consider this code experimental. This is a first pass and may not be final; it has been informally tested but awaiting rigorous testing before being signed off.
 * Also note that file writing/reading from file is not currently supported (will be added soon)
 */

package weka.classifiers.meta.timeseriesensembles;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import tsc_algorithms.cote.HiveCoteModule;
import utilities.ClassifierTools;
import utilities.CrossValidator;
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

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * James: CV pulled out to (currently) simple helper class CrossValidator
 *      results file reading added
 *      
 */

public class HESCA extends AbstractClassifier implements HiveCoteModule, SaveCVAccuracy{
    private boolean debug = false;
    
    protected final SimpleBatchFilter transform;
    protected double[] individualCvAccs;
    protected double[][] individualCvPreds;
    
    protected double[][] individualTestPreds;
    protected double[][][] individualTestDists;
    
    protected double[] ensembleCvPreds;
    protected double ensembleCvAcc;
    
    protected boolean setSeed = false;
    protected int seed;
    
    protected Classifier[] classifiers;
    protected String[] classifierNames;
    
    protected Instances train;
    
    protected boolean useResultsFileReadingWriting = false;
    protected String resultsDir;
    protected String datasetIdentifier;
    protected String ensembleIdentifier;
    protected int resampleIdentifier;
    
    
    protected boolean writeEnsembleTrainingFile = false;
    protected String outputTrainingPathAndFile;
    public HESCA(SimpleBatchFilter transform) {
        this.transform = transform;
        this.setDefaultClassifiers();
    }
    
    public HESCA() {
        this.transform = null;
        this.setDefaultClassifiers();
    }
    public HESCA(Classifier[] classifiers, String[] classifierNames) {
        this.transform = null;
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
    }
    
    public HESCA(SimpleBatchFilter transform, Classifier[] classifiers, String[] classifierNames) {
        this.transform = transform;
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
    }
    public Classifier[] getClassifiers(){ return classifiers;}
   
    public void setDebug(boolean b) { debug = b; }
    
    private void printDebug(String str) {
        if (debug)
            System.out.print(str);
    }
    
    private void printlnDebug(String str) {
        if (debug)
            System.out.println(str);
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
    }
    
    public void setRandSeed(int seed){
        this.setSeed = true;
        this.seed = seed;
    }
    
    public void turnOnResultsFileReadingWriting(String outputDir, String ensembleIdentifier, String datasetIdentifier, int resampleIdentifier){
        this.resultsDir = outputDir;
        this.ensembleIdentifier = ensembleIdentifier;
        this.datasetIdentifier = datasetIdentifier;
        this.resampleIdentifier = resampleIdentifier;
        this.useResultsFileReadingWriting = true;
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
    public void buildClassifier(Instances input) throws Exception{        
        //transform data if specified
        if(this.transform==null){
            this.train = input;
        }else{
            this.train = transform.process(input);
        }
        
        int correct;  
        this.individualCvAccs = new double[this.classifiers.length];
        this.individualCvPreds = new double[this.classifiers.length][];
        
        if (useResultsFileReadingWriting) { //train accs/preds used regardless
            //these used ONLY if test preds are being loaded 
            this.individualTestPreds = new double[this.classifiers.length][];
            this.individualTestDists = new double[this.classifiers.length][][];
        }
        
        //prep cv 
        int numFolds = findNumFolds(train);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(numFolds);
        cv.buildFolds(train);
//        this.individualCvPreds = cv.crossValidate(classifiers, input);
        
        //for each module
        for(int c = 0; c < this.classifiers.length; c++){
            
            boolean trainResultsLoaded = false;
            
            if (useResultsFileReadingWriting) {
                File moduleTrainResultsFile = findResultsFile(classifierNames[c], "train");
                if (moduleTrainResultsFile != null) { 
                    //already have acc/preds saved, load them in
                    printlnDebug(classifierNames[c] + " train loading...");
                    
                    ModulePredictions res = loadResultsFile(moduleTrainResultsFile);
                    individualCvAccs[c] = res.acc;
                    individualCvPreds[c] = res.preds;
                    
                    trainResultsLoaded = true;
                }
                
                File moduleTestResultsFile = findResultsFile(classifierNames[c], "test");
                if (moduleTestResultsFile != null) { 
                    //also load in the module's test predictions for later is present
                    //of course not actually used at all during training, only loaded for future use
                    //when classifying with ensemble
                    printlnDebug(classifierNames[c] + " test loading...");
                    
                    ModulePredictions res = loadResultsFile(moduleTestResultsFile);
                    individualTestPreds[c] = res.preds; 
                    individualTestDists[c] = res.distsForInsts; //dists not used atm, will be at later date
                }
            }
            
            if (!trainResultsLoaded) {
                printlnDebug(classifierNames[c] + " performing cv...");
                
                //no results found, do cv from scratch
                correct = 0;
                this.individualCvPreds[c] = cv.crossValidate(classifiers[c], train);
                
                //find train acc from the preds
                for(int i = 0; i < this.individualCvPreds[c].length; i++)
                    if(train.instance(i).classValue()==this.individualCvPreds[c][i])
                        correct++;
                        
                this.individualCvAccs[c] = (double)correct/train.numInstances();
            }
            
            //and do final buildclassifier on the full training data
            classifiers[c].buildClassifier(train);
            
            //write trainfold# for module if file writing and file doesnt already exist
            if(useResultsFileReadingWriting && !trainResultsLoaded){
                ModulePredictions results = new ModulePredictions(individualCvAccs[c], individualCvPreds[c], null);
                writeResultsFile(classifierNames[c], "internalHESCA", results, "train"); 
                //todo ask about parameters section, should make more informative. e.g maybe number of cv folds
                //if not the actual clasifiers parameters (that would requires some funky stuff though,
                //that's why it's not already there)
            }
        }
        
        //got module trainpreds, time to combine to find overall ensemble trainpreds 
        this.ensembleCvPreds = new double[train.numInstances()];
        
        double actual, pred;
        double bsfWeight;
        correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        
        //for each train inst
        for(int i = 0; i < train.numInstances(); i++){
            actual = train.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[train.numClasses()];

            //for each module
            for(int m = 0; m < classifiers.length; m++){
                
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
        this.ensembleCvAcc = (double)correct/train.numInstances();
        
        //if writing results of this ensemble (to be read later as an individual module of a meta ensemble, 
        //i.e cote or maybe a meta-hesca), write the full ensemble trainFold# file
        if(this.writeEnsembleTrainingFile){
            StringBuilder output = new StringBuilder();

            String hescaIdentifier = "HESCA";
            if(this.transform!=null){
                hescaIdentifier = "HESCA_"+this.transform.getClass().getSimpleName();
            }
            
            output.append(input.relationName()).append(","+hescaIdentifier+",train\n");
            output.append(this.getParameters()).append("\n");
            output.append(this.getEnsembleCvAcc()).append("\n");

            for(int i = 0; i < train.numInstances(); i++){
                output.append(train.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append("\n");
            }

            new File(this.outputTrainingPathAndFile).getParentFile().mkdirs();
            FileWriter fullTrain = new FileWriter(this.outputTrainingPathAndFile);
            fullTrain.append(output);
            fullTrain.close();
        }
        
    }

//    @Override
    public double getEnsembleCvAcc() {
        return this.ensembleCvAcc;
    }

//    @Override
    public double[] getEnsembleCvPreds() {
        return this.ensembleCvPreds;
    }

//    @Override
    public double[] getIndividualCvAccs() {
        return this.individualCvAccs;
    }

//    @Override
    public double[][] getIndividualCvPredictions() {
        return this.individualCvPreds;
    }
    
    public SimpleBatchFilter getTransform(){
        return this.transform;
    }
    
    @Override
    public void setCVPath(String pathAndName){
        this.outputTrainingPathAndFile = pathAndName;
        this.writeEnsembleTrainingFile = true;
    }     
    
    @Override
    public String getParameters(){
        StringBuilder out = new StringBuilder();
        out.append("NA,");
        for(int c = 0; c < this.classifierNames.length; c++){
            out.append(classifierNames[c]+",");
        }
        return out.toString();
    }
//
//    @Override
//    public double classifyInstance(Instance instance) throws Exception {
//        if(this.transform!=null){
//            Instances rawContainer = new Instances(template,0);
//            rawContainer.add(instance);
//            Instances converted = transform.process(rawContainer);
//            return super.classifyInstance(converted.instance(0));
//        }
//        return super.classifyInstance(instance);
//    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        //transform data if specified
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = transform.process(rawContainer);
            ins = converted.instance(0);
        }
        
        double thisPred;
        double[] preds=new double[ins.numClasses()];
        
        //get predictions for each module, turn that into a vote with the weight of 
        //that module's train acc
        for(int i=0;i<classifiers.length;i++){
            thisPred=classifiers[i].classifyInstance(ins);
            preds[(int)thisPred]+=this.individualCvAccs[i];
        }
        
        //normalise so all sum to one 
        double sum=preds[0];
        for(int i=1;i<preds.length;i++){
            sum+=preds[i];
        }
        for(int i=0;i<preds.length;i++)
            preds[i]/=sum;

        return preds;
    }
    
    /**
     * Will try to use each individual's loaded test predictions (via the testInstIndex), else will find distribution normally (via testInst)
     */
    public double[] distributionForInstance(Instance testInst, int testInstIndex) throws Exception{       
        Instance ins = testInst;
        if(this.transform!=null){
            Instances rawContainer = new Instances(testInst.dataset(),0);
            rawContainer.add(testInst);
            Instances converted = transform.process(rawContainer);
            ins = converted.instance(0);
        }
        
        int numClasses = ins.numClasses();
        double[] preds = new double[numClasses];
        double cvAccSum = 0;
        
        double pred;
        for(int c = 0; c < classifiers.length; c++){
            if (useResultsFileReadingWriting && individualTestPreds[c] != null) //if results loaded for this classifier
                pred = individualTestPreds[c][testInstIndex]; //use them
            else 
                pred = classifiers[c].classifyInstance(ins); //else find them from scratch
            
            preds[(int)pred] += this.individualCvAccs[c];
        }
        
        //normalise so all sum to one 
        double sum=preds[0];
        for(int i=1;i<preds.length;i++){
            sum+=preds[i];
        }
        for(int i=0;i<preds.length;i++)
            preds[i]/=sum;
        
        return preds;
    }
    
    /**
     * Will try to use each individual's loaded test predictions (via the testInstIndex), else will find distribution normally (via testInst)
     */
    public double classifyInstance(Instance testInst, int testInstIndex) throws Exception{     
        double[] dist = distributionForInstance(testInst, testInstIndex);
        
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
    
    public static void buildAndWriteFullIndividualTrainTestResults(Instances defaultTrainPartition, Instances defaultTestPartition, String resultOutputDir, String datasetIdentifier, String ensembleIdentifier, int resampleIdentifier, SimpleBatchFilter transform) throws Exception{
        HESCA h;
        if(transform!=null){
            h = new HESCA(transform);
        }else{
            h = new HESCA();
        }
        
        Instances train = new Instances(defaultTrainPartition);
        Instances test = new Instances(defaultTestPartition);
        if(resampleIdentifier >0){
            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleIdentifier);
            train = temp[0];
            test = temp[1];
        }
        
        
        h.turnOnResultsFileReadingWriting(resultOutputDir, ensembleIdentifier, datasetIdentifier, resampleIdentifier);
        h.setCVPath(resultOutputDir+ensembleIdentifier+"/Predictions/"+datasetIdentifier+"/trainFold"+resampleIdentifier+".csv");
        h.buildClassifier(train);
        
        StringBuilder[] byClassifier = new StringBuilder[h.classifiers.length+1];
        for(int c = 0; c < h.classifiers.length+1; c++){
            byClassifier[c] = new StringBuilder();
        }
        int[] correctByClassifier = new int[h.classifiers.length+1];
        
        double cvSum = 0;
        for(int c = 0; c < h.classifiers.length; c++){
            cvSum+= h.individualCvAccs[c];
        }
        int correctByEnsemble = 0;
                
        
        
        double act;
        double pred;
        double[] preds;
        
        double[] distForIns = null;
        double bsfClassVal = -1;
        double bsfClassWeight = -1;
        for(int i = 0; i < test.numInstances(); i++){
            act = test.instance(i).classValue();
            preds = h.classifyInstanceByConstituents(test.instance(i));
            distForIns = new double[test.numClasses()];
            bsfClassVal = -1;
            bsfClassWeight = -1;
            for(int c = 0; c < h.classifiers.length; c++){
                byClassifier[c].append(act+","+preds[c]+"\n");
                if(preds[c]==act){
                    correctByClassifier[c]++;
                }
                
                distForIns[(int)preds[c]]+= h.individualCvAccs[c];
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
        
        FileWriter out;
        String outPath;
        for(int c = 0; c < h.classifiers.length; c++){
            outPath = h.resultsDir+h.ensembleIdentifier+h.classifierNames[c]+"/Predictions/"+h.datasetIdentifier;
            new File(outPath).mkdirs();
            out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
            out.append(h.datasetIdentifier+","+h.ensembleIdentifier+h.classifierNames[c]+",test\n");
            out.append("noParamInfo\n");
            out.append((double)correctByClassifier[c]/test.numInstances()+"\n");
            out.append(byClassifier[c]);
            out.close();            
        }
        
        outPath = h.resultsDir+h.ensembleIdentifier+"/Predictions/"+h.datasetIdentifier;
        new File(outPath).mkdirs();
        out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
        out.append(h.datasetIdentifier+","+h.ensembleIdentifier+",test\n");
        out.append("noParamInfo\n");
        out.append((double)correctByEnsemble/test.numInstances()+"\n");
        out.append(byClassifier[h.classifiers.length]);
        out.close(); 
        
    }
    
    
    
    public static void main(String[] args) throws Exception{
        Instances train = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
        
        buildAndWriteFullIndividualTrainTestResults(train, test, "hescatest/", "ItalyPowerDemand", "", 0, null);
        HESCA h = new HESCA();
        h.setRandSeed(0);
        h.setDebug(true);
        h.turnOnResultsFileReadingWriting("hescatest/", "htest", "ItalyPowerDemand", 0);
        h.buildClassifier(train);
        
        double correct = 0;
        for (int i = 0; i < test.numInstances(); ++i) {
            double pred = h.classifyInstance(test.get(i), i);
            if (pred == test.get(i).classValue())
                correct++;
        }
        
        System.out.println("\n acc=" + (correct/test.numInstances()));
        
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
    
    public static class ModulePredictions { 
        public double[][] distsForInsts;
        public double[] preds;
        public double acc; 

        public ModulePredictions(double acc, double[] preds, double[][] distsForInsts) {
            this.preds = preds;
            this.acc = acc;
            this.distsForInsts = distsForInsts;
        }
    }         
    
    public File findResultsFile(String classifierName, String trainOrTest) {
        File file = new File(resultsDir+ensembleIdentifier+classifierName+"/Predictions/"+datasetIdentifier+"/"+trainOrTest+"Fold"+resampleIdentifier+".csv");
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
                for (int i = 3; i < 3+train.numClasses(); ++i)  //act,pred,[empty],firstclassprob.... therefore 3 start
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
    
    /**
     * @param parameters for now, seems to just be 'internalHESCA', added as param for 
     * potential future use
     */
    private void writeResultsFile(String classifierName, String parameters, ModulePredictions results, String trainOrTest) throws IOException {
        printlnDebug(classifierName + " " + trainOrTest + " writing...");
        
        StringBuilder st = new StringBuilder();
        st.append(this.datasetIdentifier).append(",").append(this.ensembleIdentifier).append(classifierName).append(","+trainOrTest+"\n");
        st.append(parameters + "\n"); //st.append("internalHesca\n");
        st.append(results.acc).append("\n");
        
        for(int i = 0; i < results.preds.length;i++) {
            st.append(train.instance(i).classValue()).append(",").append(results.preds[i]).append(","); //pred
            
            if (results.distsForInsts != null && results.distsForInsts[i] != null)
                for (int j = 0; j < results.distsForInsts[i].length; j++)
                    st.append("," + results.distsForInsts[i][j]);
            
            st.append("\n");
        }
        
        String fullPath = this.resultsDir+this.ensembleIdentifier+classifierName+"/Predictions/"+datasetIdentifier;
        new File(fullPath).mkdirs();
        FileWriter out = new FileWriter(fullPath+"/" + trainOrTest + "Fold"+this.resampleIdentifier+".csv");
        out.append(st);
        out.close();
    }
    
}
