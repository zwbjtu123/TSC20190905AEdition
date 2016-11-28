/**
 * NOTE: consider this code experimental. This is a first pass and may not be final; it has been informally tested but awaiting rigorous testing before being signed off.
 * Also note that file writing/reading from file is not currently supported (will be added soon)
 */

package weka.classifiers.meta.timeseriesensembles;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import tsc_algorithms.cote.HiveCoteModule;
import utilities.ClassifierTools;
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
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;
import utilities.SaveCVAccuracy;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Needs documentation and comments
 */

// NOTE: this version doesn't currently support file writing (to-do)

public class HESCA extends AbstractClassifier implements HiveCoteModule, SaveCVAccuracy{
    protected final SimpleBatchFilter transform;
    protected double[] individualCvAccs;
    protected double[][] individualCvPreds;
    protected double[] ensembleCvPreds;
    protected double ensembleCvAcc;
    
    protected boolean setSeed = false;
    protected int seed;
    
    protected Classifier[] classifiers;
    protected String[] classifierNames;
    
    protected Instances train;
    
    // used for file writing nameing only
    protected boolean writeIndividualClassifierOutputs = false;
    protected String outputResultsDir;
    protected String datasetIdentifier;
    protected String ensembleIdentifier;
    protected int resampleIdentifier;
    
    
    protected boolean writeTraining = false;
    protected String outputTrainingPathAndFile;
    public HESCA(SimpleBatchFilter transform) {
        this.transform = transform;
        this.setDefaultClassifiers();
    }
    
    public HESCA() {
        this.transform = null;
        this.setDefaultClassifiers();
    }
    
    public HESCA(SimpleBatchFilter transform, Classifier[] classifiers, String[] classifierNames) {
        this.transform = transform;
        this.classifiers = classifiers;
        this.classifierNames = classifierNames;
    }
   public Classifier[] getClassifiers(){ return classifiers;}
   
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
    
    public void turnOnIndividualClassifierOutputs(String outputDir, String ensembleIdentifier, String datasetIdentifier, int resampleIdentifier){
        this.outputResultsDir = outputDir;
        this.ensembleIdentifier = ensembleIdentifier;
        this.datasetIdentifier = datasetIdentifier;
        this.resampleIdentifier = resampleIdentifier;
        this.writeIndividualClassifierOutputs = true;
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
      
    
/**
 * this method is very memory intensive
 * 
 */    
    public double[] crossValidate(Classifier classifier, Instances train) throws Exception{
        
        int numFolds = findNumFolds(train);
               
        Random r = null;
        if(this.setSeed){
            r = new Random(this.seed);
        }else{
            r = new Random();
        }
        ArrayList<Instances> folds = new ArrayList<>();
        ArrayList<ArrayList<Integer>> foldIndexing = new ArrayList<>();
        
        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(train,0));
            foldIndexing.add(new ArrayList<>());
        }
        
        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < train.numInstances(); i++){
            instanceIds.add(i);
        }
        Collections.shuffle(instanceIds, r);
        
        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < train.numClasses(); i++){
            byClass.add(new Instances(train,0));
            byClassIndices.add(new ArrayList<>());
        }
        
        int thisInstanceId;
        double thisClassVal;
        for(int i = 0; i < train.numInstances(); i++){
            thisInstanceId = instanceIds.get(i);
            thisClassVal = train.instance(thisInstanceId).classValue();
            
            byClass.get((int)thisClassVal).add(train.instance(thisInstanceId));
            byClassIndices.get((int)thisClassVal).add(thisInstanceId);
        }

         // now stratify        
        Instances strat = new Instances(train,0);
        ArrayList<Integer> stratIndices = new ArrayList<>();
        int stratCount = 0;
        int[] classCounters = new int[train.numClasses()];
        
        while(stratCount < train.numInstances()){
            
            for(int c = 0; c < train.numClasses(); c++){
                if(classCounters[c] < byClass.get(c).size()){
                    strat.add(byClass.get(c).instance(classCounters[c]));
                    stratIndices.add(byClassIndices.get(c).get(classCounters[c]));
                    classCounters[c]++;
                    stratCount++;
                }
            }
        }
        

        train = strat;
        instanceIds = stratIndices;
       
        double foldSize = (double)train.numInstances()/numFolds;
        
        double thisSum = 0;
        double lastSum = 0;
        int floor;
        int foldSum = 0;
        

        int currentStart = 0;
        for(int f = 0; f < numFolds; f++){

            
            thisSum = lastSum+foldSize+0.000000000001;  
// to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
            floor = (int)thisSum;
            
            if(f==numFolds-1){
                floor = train.numInstances(); // to make sure all instances are allocated in case of double imprecision causing one to go missing
            }
            
            for(int i = currentStart; i < floor; i++){
                folds.get(f).add(train.instance(i));
                foldIndexing.get(f).add(instanceIds.get(i));
            }

            foldSum+=(floor-currentStart);
            currentStart = floor;
            lastSum = thisSum;
        }
        
        if(foldSum!=train.numInstances()){
            throw new Exception("Error! Some instances got lost file creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
        }
        Instances trainCV;
        Instances testCV;
        
        double pred;
        double[] predictions = new double[train.numInstances()];
        
        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake
        
        for(int testFold = 0; testFold < numFolds; testFold++){
            
            trainCV = null;
            testCV = new Instances(folds.get(testFold));
            
            for(int f = 0; f < numFolds; f++){
                if(f==testFold){
                    continue;
                }
                temp = new Instances(folds.get(f));
                if(trainCV==null){
                    trainCV = temp;
                }else{
                    trainCV.addAll(temp);
                }
            }
            
            classifier.buildClassifier(trainCV);
            for(int i = 0; i < testCV.numInstances(); i++){
                pred = classifier.classifyInstance(testCV.instance(i));
                predictions[foldIndexing.get(testFold).get(i)] = pred;
            }
        }
        return predictions;
    }

    
    @Override
    public void buildClassifier(Instances input) throws Exception{
        if(this.transform==null){
            this.train = input;
        }else{
            this.train = transform.process(input);
        }
        
        int correct;  
        this.individualCvPreds = new double[this.classifiers.length][this.train.numInstances()];
        this.individualCvAccs = new double[this.classifiers.length];
        
        for(int c = 0; c < this.classifiers.length; c++){
//            if(classifiers[c] instanceof BaggedRandomForest){
// Implement after testing whether 10x fold is as accurate             
/*      Need to recover the CVPredictions for RandomForest using Enhanced RandomForest    
            if(classifiers[c] instanceof RandomForest){
                classifiers[c].buildClassifier(train);
                individualCvAccs[c]=1-((RandomForest)classifiers[c]).measureOutOfBagError();                
            }
            else 
*/
            {
                this.individualCvPreds[c] = crossValidate(this.classifiers[c],this.train);
                correct = 0;

                for(int i = 0; i < this.individualCvPreds[c].length; i++){
                    if(train.instance(i).classValue()==this.individualCvPreds[c][i]){
                        correct++;
                    }
                }
                this.individualCvAccs[c] = (double)correct/train.numInstances();
                classifiers[c].buildClassifier(train);
            }
            if(this.writeIndividualClassifierOutputs){
                StringBuilder st = new StringBuilder();
                st.append(this.datasetIdentifier+","+this.ensembleIdentifier+"_"+classifierNames[c]+",train\n");
                st.append("internalHesca\n");
                st.append(individualCvAccs[c]+"\n");
                for(int i = 0; i < this.individualCvPreds[c].length;i++){
                    st.append(train.instance(i).classValue()+","+individualCvPreds[c][i]+"\n");
                }
                new File(this.outputResultsDir+this.ensembleIdentifier+"_"+classifierNames[c]+"/Predictions/"+datasetIdentifier).mkdirs();
                FileWriter out = new FileWriter(this.outputResultsDir+this.ensembleIdentifier+"_"+classifierNames[c]+"/Predictions/"+datasetIdentifier+"/trainFold"+this.resampleIdentifier+".csv");
                out.append(st);
                out.close();
            }
            
            
            
            
        }
        
        this.ensembleCvPreds = new double[train.numInstances()];
        
        double actual, pred;
        double bsfWeight;
        correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        for(int i = 0; i < train.numInstances(); i++){
            actual = train.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[train.numClasses()];

            for(int m = 0; m < classifiers.length; m++){
                weightByClass[(int)individualCvPreds[m][i]]+=this.individualCvAccs[m];
                
                if(weightByClass[(int)individualCvPreds[m][i]] > bsfWeight){
                    bsfWeight = weightByClass[(int)individualCvPreds[m][i]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(individualCvPreds[m][i]);
                }else if(weightByClass[(int)individualCvPreds[m][i]] == bsfWeight){
                    bsfClassVals.add(individualCvPreds[m][i]);
                }
            }
            
            if(bsfClassVals.size()>1){
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            }else{
                pred = bsfClassVals.get(0);
            }
            
            if(pred==actual){
                correct++;
            }
            this.ensembleCvPreds[i]=pred;
        }
        this.ensembleCvAcc = (double)correct/train.numInstances();
        
        if(this.writeTraining){
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
        this.writeTraining = true;
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
//    public double[] distributionForInstance(Instance instance) throws Exception {
//        if(this.transform!=null){
//            Instances rawContainer = new Instances(template,0);
//            rawContainer.add(instance);
//            Instances converted = transform.process(rawContainer);
//            return super.distributionForInstance(converted.instance(0));
//        }
//        return super.distributionForInstance(instance);
//    }
    public double[] distributionForInstance(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            Instances converted = transform.process(rawContainer);
            ins = converted.instance(0);
        }
        
        double thisPred;
        double[] preds=new double[ins.numClasses()];
        
        
        for(int i=0;i<classifiers.length;i++){
            thisPred=classifiers[i].classifyInstance(ins);
            preds[(int)thisPred]+=this.individualCvAccs[i];
        }
        double sum=preds[0];
        for(int i=1;i<preds.length;i++){
            sum+=preds[i];
        }
        for(int i=0;i<preds.length;i++)
            preds[i]/=sum;

        
        return preds;
    }
    
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
        
        
        h.turnOnIndividualClassifierOutputs(resultOutputDir, ensembleIdentifier, datasetIdentifier, resampleIdentifier);
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
            outPath = h.outputResultsDir+h.ensembleIdentifier+"_"+h.classifierNames[c]+"/Predictions/"+h.datasetIdentifier;
            new File(outPath).mkdirs();
            out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
            out.append(h.datasetIdentifier+","+h.ensembleIdentifier+"_"+h.classifierNames[c]+",test\n");
            out.append("noParamInfo\n");
            out.append((double)correctByClassifier[c]/test.numInstances()+"\n");
            out.append(byClassifier[c]);
            out.close();            
        }
        
        outPath = h.outputResultsDir+h.ensembleIdentifier+"/Predictions/"+h.datasetIdentifier;
        new File(outPath).mkdirs();
        out = new FileWriter(outPath+"/testFold"+h.resampleIdentifier+".csv");
        out.append(h.datasetIdentifier+","+h.ensembleIdentifier+",test\n");
        out.append("noParamInfo\n");
        out.append((double)correctByEnsemble/test.numInstances()+"\n");
        out.append(byClassifier[h.classifiers.length]);
        out.close(); 
        
    }
    
    
    
    public static void main(String[] args) throws Exception{
        Instances train = ClassifierTools.loadData("c:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
        Instances test = ClassifierTools.loadData("c:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
        ShapeletTransform st = ShapeletTransformFactory.createTransform(train);
        HESCA th = new HESCA(st);
//        EnhancedHESCA th = new EnhancedHESCA();
        th.buildClassifier(train);
//        System.out.println(th.getEnsembleCvAcc());
//        double[] individualCvs = th.getIndividualCvAccs();
//        for(double acc:individualCvs){
//            System.out.print(acc+",");
//        }
        
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){
            if(th.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
            System.out.println(th.classifyInstance(test.instance(i))+"\t"+test.instance(i).classValue());
        }
        System.out.println(correct+"/"+test.numInstances());
        System.out.println((double)correct/test.numInstances());
    }
}
