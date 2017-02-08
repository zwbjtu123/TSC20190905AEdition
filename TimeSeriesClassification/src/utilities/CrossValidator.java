/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;

/**
 * Start of a custom cross validation class, to be built on/optimised over time as
 * work with ensembles progresses
 * 
 * Initial push uses Jay's stratified folding code from HESCA
 * 
 * @author James 
 */
public class CrossValidator {
            
    private Integer seed = null;
    private int numFolds;
    private ArrayList<Instances> folds;
    private ArrayList<ArrayList<Integer>> foldIndexing;

    public CrossValidator() {
        this.seed = null;
        this.folds = null;
        this.foldIndexing = null;
        this.numFolds = 10;
    }

    public Integer getSeed() {
        return seed;
    }

    public void setSeed(Integer seed) {
        this.seed = seed;
    }

    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }

    /**
     * @return the index in the original train set of the instance found at folds.get(fold).get(indexInFold) 
     */
    public int getOriginalInstIndex(int fold, int indexInFold) {
        return foldIndexing.get(fold).get(indexInFold);
    }

    public double[] crossValidate(Classifier classifier, Instances train) throws Exception{
        return crossValidate(new Classifier[] { classifier }, train)[0];
    }

    /**
     * Crossvalidates all classifiers provided using the same fold split for all,
     * i.e for each prediction, all classifiers will have trained on the exact same
     * subset data to have made that classification
     * 
     * If folds have already been defined (by a call to buildFolds()), will use those,
     * else will create them internally 
     * 
     * @return double[classifier][prediction]
     */
    public double[][] crossValidate(Classifier[] classifiers, Instances train) throws Exception{
        if (folds == null)
            buildFolds(train);

        double pred;
        double[][] predictions = new double[classifiers.length][train.numInstances()];

        //for each fold as test
        for(int testFold = 0; testFold < numFolds; testFold++){
            Instances[] trainTest = buildTrainTestSet(testFold);

            //for each classifier in ensemble
            for (int c = 0; c < classifiers.length; ++c) {
                classifiers[c].buildClassifier(trainTest[0]);

                //for each test instance on this fold
                for(int i = 0; i < trainTest[1].numInstances(); i++){
                    //classify and store prediction
                    pred = classifiers[c].classifyInstance(trainTest[1].instance(i));
                    predictions[c][getOriginalInstIndex(testFold, i)] = pred;
                }    
            }
        }
        return predictions;
    }

    /**
     * @return [0] = new train set, [1] = test(validation) set
     */
    public Instances[] buildTrainTestSet(int testFold) {
        Instances[] trainTest = new Instances[2];
        trainTest[0] = null;
        trainTest[1] = new Instances(folds.get(testFold));

        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake
        for(int f = 0; f < folds.size(); f++){
            if(f==testFold){
                continue;
            }
            temp = new Instances(folds.get(f));
            if(trainTest[0]==null){
                trainTest[0] = temp;
            }else{
                trainTest[0].addAll(temp);
            }
        }

        return trainTest;
    }

    public void buildFolds(Instances train) throws Exception {               
        Random r = null;
        if(seed != null){
            r = new Random(seed);
        }else{
            r = new Random();
        }

        folds = new ArrayList<Instances>();
        foldIndexing = new ArrayList<ArrayList<Integer>>();

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
            throw new Exception("Error! Some instances got lost while creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
        }
    }
    
    
    public static void main(String[] args) throws Exception {
        CrossValidator cv = new CrossValidator();
        cv.setNumFolds(10);
        cv.setSeed(0);
        
        Classifier c = new kNN();
        Instances insts = ClassifierTools.loadData("C:/TSC Problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
        
        double[] preds = cv.crossValidate(c, insts);
        
        double acc = 0.0;
        System.out.println("Pred | Actual");
        for (int i = 0; i < preds.length; i++) {
            System.out.printf("%4d | %d\n", (int)preds[i], (int)insts.get(i).classValue());
            if (preds[i] == insts.get(i).classValue())
                ++acc;
        }
        
        acc /= preds.length;
        System.out.println("\n Acc: " + acc);
    }
}
