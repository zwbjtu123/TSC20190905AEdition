
package weka.classifiers.meta.timeseriesensembles;

import java.text.DecimalFormat;
import java.util.ArrayList;

/**
 * Simple container class for the results of a classifier (module) on a dataset
 * It can be used in batch mode (add them all in at once) or online (add in one at a time).
 * The former is sensible for storing training set results, the latter for test results
 * 
 * 
 * @author James Large
 */
public class ClassifierResults {
    public long buildTime;
    public double acc; 
    public double stddev; //across cv folds
    private int numClasses;

    public double[][] confusionMatrix; //[actual class][predicted class]
    

    
    public ArrayList<Double> actualClassValues;
    public ArrayList<Double> predictedClassValues;
    public ArrayList<double[]> predictedClassProbabilities;
    
    /**
     * for building results one by one while testing, call finaliseResults
 to populate the ClassifierResults object once testing is finished
     */
    public ClassifierResults() {
        actualClassValues= new ArrayList<>();
        predictedClassValues = new ArrayList<>();
        predictedClassProbabilities = new ArrayList<>();
    }
    
    //for if we are only storing the cv accuracy in the context of SaveCVAccuracy
    public ClassifierResults(double cvacc, int numClasses) {
        this();
        this.acc = cvacc;
        this.numClasses = numClasses;
    }
    
    public ClassifierResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, int numClasses) {        
        this();
        for(double d:preds)
            predictedClassValues.add(d);
        this.acc = acc;
        for(double[] d:distsForInsts)
            predictedClassProbabilities.add(d);
 
        this.numClasses = numClasses;
       for(double d:classVals)
           actualClassValues.add(d);
        this.confusionMatrix = buildConfusionMatrix();
        
        this.stddev = -1; //not defined 
    }
    
    public ClassifierResults(double acc, double[] classVals, double[] preds, double[][] distsForInsts, double stddev, int numClasses) { 
        this(acc,classVals,preds,distsForInsts,numClasses);
        this.stddev = stddev; 
    }

    /**
    * @return [actual class][predicted class]
    */
    private double[][] buildConfusionMatrix() {
        double[][] matrix = new double[numClasses][numClasses];
        for (int i = 0; i < predictedClassValues.size(); ++i){
            double actual=actualClassValues.get(i);
            double predicted=predictedClassValues.get(i);
            ++matrix[(int)actual][(int)predicted];
        }
        return matrix;
    }
    public void addAllResults(double[] classVals, double[] preds, double[][] distsForInsts, int numClasses){
//Overwrites previous        
        actualClassValues= new ArrayList<>();
        predictedClassValues = new ArrayList<>();
        predictedClassProbabilities = new ArrayList<>();
        for(double d:preds)
            predictedClassValues.add(d);
        this.acc = acc;
        for(double[] d:distsForInsts)
            predictedClassProbabilities.add(d);
 
        this.numClasses = numClasses;
       for(double d:classVals)
           actualClassValues.add(d);
        this.confusionMatrix = buildConfusionMatrix();
        
        this.stddev = -1; //not defined 
        
    }
 //Pass the probability estimates for each class, but need the true class too
    public void storeSingleResult(double[] dist) {        
        double max = dist[0];
        double maxInd = 0;
        predictedClassProbabilities.add(dist);
        for (int i = 0; i < dist.length; i++) {
            if (dist[i] > max) {
                max = dist[i];
                maxInd = i;
            }
        }
        predictedClassValues.add(maxInd);
    }
    
    
    public void finaliseResults(double[] testClassVals) throws Exception {
        if (predictedClassProbabilities == null || predictedClassValues == null ||
                predictedClassProbabilities.isEmpty() || predictedClassValues.isEmpty())
            throw new Exception("finaliseTestResults(): no test predictions stored for this module");
        
        if (testClassVals.length != predictedClassValues.size())
            throw new Exception("finaliseTestResults(): Number of test predictions made and number of test cases do not match");
        
        for(double d:testClassVals)
            actualClassValues.add(d);
        
        
        double correct = .0;
        for (int inst = 0; inst < predictedClassValues.size(); inst++) {
            if (testClassVals[inst] == predictedClassValues.get(inst))
                    ++correct;
        }
        acc = correct/testClassVals.length;
    }
    public int numInstances(){ return predictedClassValues.size();}
    public double[] getTrueClassVals(){
        double[] d=new double[actualClassValues.size()];
        int i=0;
        for(double x:actualClassValues)
            d[i++]=x;
        return d;
    }
   public double[] getPredClassVals(){
        double[] d=new double[predictedClassValues.size()];
        int i=0;
        for(double x:predictedClassValues)
            d[i++]=x;
        return d;
    }
   public double getPredClassValue(int index){
        return predictedClassValues.get(index);
    }
   public double getTrueClassValue(int index){
        return actualClassValues.get(index);
    }
   public double[] getDistributionForInstance(int i){
       if(i<predictedClassProbabilities.size())
            return predictedClassProbabilities.get(i);
       return null;
   }
   public String writeInstancePredictions(){
       DecimalFormat df=new DecimalFormat("#.######");
       if(numInstances()>0 &&(predictedClassProbabilities.size()==actualClassValues.size()&& predictedClassProbabilities.size()==predictedClassValues.size())){
           StringBuilder sb=new StringBuilder("");
           for(int i=0;i<numInstances();i++){
               sb.append(actualClassValues.get(i).intValue()).append(",");
               sb.append(predictedClassValues.get(i).intValue()).append(",");
               double[] probs=predictedClassProbabilities.get(i);
               for(double d:probs)
                   sb.append(",").append(df.format(d));
               sb.append("\n");
           }
           if(confusionMatrix==null)
               confusionMatrix=buildConfusionMatrix();
           for(int i=0;i<confusionMatrix.length;i++){
               for(int j=0;j<confusionMatrix[i].length;i++){
                   sb.append(confusionMatrix[i][j]);
                   if(j<confusionMatrix[i].length-1)
                       sb.append(",");
               }
               sb.append("\n");
           }
           return sb.toString();
       }
       else
           return "No Instance Prediction Information";
       
   }
   boolean hasInstanceData(){ return numInstances()!=0;}
   
}
