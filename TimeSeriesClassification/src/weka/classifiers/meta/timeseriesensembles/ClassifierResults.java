
package weka.classifiers.meta.timeseriesensembles;

import java.util.ArrayList;

/**
 * Simple container class for the results of a classifier (module) on a dataset
 * 
 * @author James Large
 */
public class ClassifierResults {
    public double[] predClassVals;
    public double acc; 
    public long buildTime;
    public double[][] distsForInsts; 
    public double[][] confusionMatrix; //[actual class][predicted class]
    public double[] trueClassVals;
    public double stddev; //across cv folds
    
    private int numClasses;

    //todo, re-work how this is done
    //turns out storing/writing the test preds within the class that's being tested
    //is extremely annoying and doesnt fit with the rest of the class structure
    private ArrayList<Double> testPredsSoFar;
    private ArrayList<ArrayList<Double>> testDistsSoFar;
    
    /**
     * for building results one by one while testing, call finaliseTestResults
     * to populate the ClassifierResults object once testing is finished
     */
    public ClassifierResults() {
        testPredsSoFar = new ArrayList<>();
        testDistsSoFar = new ArrayList<>();
    }
    
    //for if we are only storing the cv accuracy in the context of SaveCVAccuracy
    public ClassifierResults(double cvacc, int numClasses) {
        this.acc = cvacc;
        this.numClasses = numClasses;
    }
    
    public ClassifierResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, int numClasses) {        
        this.predClassVals = preds;
        this.acc = acc;
        this.distsForInsts = distsForInsts;

        this.numClasses = numClasses;

        this.trueClassVals = classVals;
        this.confusionMatrix = buildConfusionMatrix();
        
        this.stddev = -1; //not defined 
    }
    
    public ClassifierResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, double stddev, int numClasses) {        
        this.predClassVals = preds;
        this.acc = acc;
        this.distsForInsts = distsForInsts;

        this.numClasses = numClasses;

        this.trueClassVals = classVals;
        this.confusionMatrix = buildConfusionMatrix();
        
        this.stddev = stddev; 
    }

    /**
    * @return [actual class][predicted class]
    */
    private double[][] buildConfusionMatrix() {
        double[][] matrix = new double[numClasses][numClasses];
        for (int i = 0; i < predClassVals.length; ++i)
            ++matrix[(int)trueClassVals[i]][(int)predClassVals[i]];

        return matrix;
    }
    
    //again when i have the willpower find a better way to do this
    public void storeSingleTestResult(double[] dist) {        
        double max = dist[0];
        double maxInd = 0;
           
        ArrayList<Double> a = new ArrayList<>();
        for (int i = 0; i < dist.length; i++) {
            a.add(dist[i]);
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        
        testDistsSoFar.add(a);
        testPredsSoFar.add(maxInd);
    }
    
    
    public void finaliseTestResults(double[] testClassVals) throws Exception {
        if (testDistsSoFar == null || testPredsSoFar == null ||
                testDistsSoFar.size() == 0 || testPredsSoFar.size() == 0)
            throw new Exception("finaliseTestResults(): no test predictions stored for this module");
        
        if (testClassVals.length != testPredsSoFar.size())
            throw new Exception("finaliseTestResults(): Number of test predictions made and number of test cases do not match");
        
        this.trueClassVals = testClassVals;
        
        distsForInsts = new double[testDistsSoFar.size()][testDistsSoFar.get(0).size()];
        predClassVals = new double[testDistsSoFar.size()];
        
        double correct = .0;
        for (int inst = 0; inst < distsForInsts.length; inst++) {
            predClassVals[inst] = testPredsSoFar.get(inst);
            if (testClassVals[inst] == predClassVals[inst])
                    ++correct;
            
            for (int c = 0; c < distsForInsts[inst].length; c++) 
                distsForInsts[inst][c] = testDistsSoFar.get(inst).get(c);
        }
        
        acc = correct/testClassVals.length;
    }
}
