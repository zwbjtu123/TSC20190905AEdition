/**
 * NOTE: consider this code experimental. This is a first pass and may not be final; it has been informally tested but awaiting rigurous testing before being signed off.
 * Also note that file writing/reading from file is not currently supported (will be added soon)
 */

package weka.classifiers.meta.timeseriesensembles;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import tsc_algorithms.cote.HiveCoteModule;
//import tsc_algorithms.cote.HiveCoteModule;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.depreciated.HESCA_05_10_16;
import weka.classifiers.trees.EnhancedRandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */

// NOTE: this version doesn't currently support file writing (to-do)

public class HESCA extends AbstractClassifier implements HiveCoteModule{
//public class HESCA extends AbstractClassifier {

    private final SimpleBatchFilter transform;
    private double[] individualCvAccs;
    private double[][] individualCvPreds;
    private double[] ensembleCvPreds;
    private double ensembleCvAcc;
    
    private boolean setSeed = false;
    private int seed;
    
    private Classifier[] classifiers;
    private String[] classifierNames;
    
    private Instances train;
    
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
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        svml.setKernel(kl);
        if(setSeed)
            svml.setRandomSeed(seed);
        classifiers[3] = svml;
        classifierNames[3] = "SVML";
        
        SMO svmq =new SMO();
        PolyKernel kq = new PolyKernel();
        kq.setExponent(2);
        svmq.setKernel(kq);
        if(setSeed)
           svmq.setRandomSeed(seed);
        classifiers[4] =svmq;
        classifierNames[4] = "SVMQ";
        
        RandomForest r=new EnhancedRandomForest();
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
    
    public void setSeed(int seed){
        this.setSeed = true;
        this.seed = seed;
    }
    
    public double[] crossValidate(Classifier classifier, Instances train) throws Exception{
        
        int numFolds = train.numInstances();
        if(numFolds>HESCA_05_10_16.MAX_NOS_FOLDS){
            numFolds = HESCA_05_10_16.MAX_NOS_FOLDS;
        }      
               
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

            
            thisSum = lastSum+foldSize+0.000000000001;  // to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
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
        

        Instances trainLoocv;
        Instances testLoocv;
        
        double pred, actual;
        double[] predictions = new double[train.numInstances()];
        
        int correct = 0;
        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake
        
        for(int testFold = 0; testFold < numFolds; testFold++){
            
            trainLoocv = null;
            testLoocv = new Instances(folds.get(testFold));
            
            for(int f = 0; f < numFolds; f++){
                if(f==testFold){
                    continue;
                }
                temp = new Instances(folds.get(f));
                if(trainLoocv==null){
                    trainLoocv = temp;
                }else{
                    trainLoocv.addAll(temp);
                }
            }
            
            classifier.buildClassifier(trainLoocv);
            for(int i = 0; i < testLoocv.numInstances(); i++){
                pred = classifier.classifyInstance(testLoocv.instance(i));
                actual = testLoocv.instance(i).classValue();
                predictions[foldIndexing.get(testFold).get(i)] = pred;
                if(pred==actual){
                    correct++;
                }
            }
        }
        
//        double acc = (double)correct/train.numInstances();
        return predictions;
    }

    
    @Override
    public void buildClassifier(Instances input) throws Exception{
//        template = new Instances(input,0);
        if(this.transform==null){
            this.train = input;
        }else{
            this.train = transform.process(input);
        }
        
        int correct;  
        this.individualCvPreds = new double[this.classifiers.length][this.train.numInstances()];
        this.individualCvAccs = new double[this.classifiers.length];
        
        for(int i = 0; i < this.classifiers.length; i++){
            this.individualCvPreds[i] = crossValidate(this.classifiers[i],this.train);
            correct = 0;
            
            for(int j = 0; j < this.individualCvPreds[i].length; j++){
                if(train.instance(j).classValue()==this.individualCvPreds[i][j]){
                    correct++;
                }
            }
            this.individualCvAccs[i] = (double)correct/train.numInstances();
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
    }

//    @Override
    public double getEnsembleCvAcc() {
        return this.ensembleCvAcc;
    }

//    @Override
    public double[] getEnsembleCvPredictions() {
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
