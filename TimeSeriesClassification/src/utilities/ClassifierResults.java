
package utilities;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import timeseriesweka.classifiers.ensembles.weightings.FScore;
import utilities.generic_storage.Pair;

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
    private int numClasses;
    private int numInstances;
    private String name;
    private String paras;
    
    public double acc; 
    public double balancedAcc; 
    public double sensitivity;
    public double specificity;
    public double precision;
    public double recall;
    
    public double f1; 
    public double nll; 
    public double stddev; //across cv folds
    
    public double[][] confusionMatrix; //[actual class][predicted class]
    public double[] countPerClass;

    
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
    public void storeSingleResult(double actual, double[] dist) {        
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
        actualClassValues.add(actual);
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
               if(i<numInstances()-1)
                   sb.append("\n");
           }
           return sb.toString();
       }
       else
           return "No Instance Prediction Information";
       
   }
   public void loadFromFile(String path) throws FileNotFoundException{
       File f=new File(path);
       if(f.exists() && f.length()>0){
           InFile inf=new InFile(path);
           name = inf.readLine();
           paras= inf.readLine();
           double testAcc=inf.readDouble();
           String line=inf.readLine();
            actualClassValues= new ArrayList<>();
            predictedClassValues = new ArrayList<>();
            predictedClassProbabilities = new ArrayList<>();
            numInstances=0;
            acc=0;
           while(line!=null){
               String[] split=line.split(",");
               if(split.length>3){
                    double a=Double.valueOf(split[0]);
                    double b=Double.valueOf(split[1]);
                    actualClassValues.add(a);
                    predictedClassValues.add(b);
                    if(a==b)
                        acc++;
                    if(numInstances==0){
                        numClasses=split.length-3;   //Check!
                    }
                    double[] probs=new double[numClasses];
                    for(int i=0;i<probs.length;i++)
                        probs[i]=Double.valueOf(split[3+i]);
                    predictedClassProbabilities.add(probs);
                   numInstances++;
               }
               line=inf.readLine();
           }
           acc/=numInstances;
       }
       else
           throw new FileNotFoundException("File "+path+" NOT FOUND");
   }
/**
 * Find: Accuracy, Balanced Accuracy, F1 (1 vs All averaged?), 
 * Sensitivity, Specificity, AUROC, negative log likelihood  
 */   
   public void findAllStats(){
       confusionMatrix=buildConfusionMatrix();
       countPerClass=new double[confusionMatrix.length];
       for(int i=0;i<actualClassValues.size();i++)
           countPerClass[actualClassValues.get(i).intValue()]++;
//Accuracy       
       acc=0;
       for(int i=0;i<numClasses;i++)
           acc+=confusionMatrix[i][i];
       acc/=numInstances;
//Balanced accuracy
       balancedAcc=findBalancedAcc(confusionMatrix);
//F1
       f1=findF1(confusionMatrix);
//Negative log likelihood
       nll=findNLL();
           
       }
       public double findNLL(){
           double nll=0;
            for(int i=0;i<actualClassValues.size();i++){
                double[] dist=getDistributionForInstance(i);
           
            }
            return nll;
   }
   
/**
 * Balanced accuracy: average of the accuracy for each class
 * @param cm
 * @return 
 */   
    public double findBalancedAcc(double[][] cm){
       double[] accPerClass=new double[cm.length];
       for(int i=0;i<cm.length;i++)
           accPerClass[i]=cm[i][i]/countPerClass[i];
       double b=accPerClass[0];
       for(int i=1;i<cm.length;i++)
          b+=accPerClass[i]; 
       return b;
    }
 /**
  * F1: If it is a two class problem we use the minority class
  * if it is multiclass we average over all classes. 
  * @param cm
  * @return 
  */   
    public double findF1(double[][] cm){
        double f=0;
        if(numClasses==2){
            if(countPerClass[0]<countPerClass[1])
                f=computeFScore(cm,0,1);
            else
                f=computeFScore(cm,1,1);
        }
        else{//Average over all of them
            for(int i=0;i<numClasses;i++)
                f+=computeFScore(cm,i,1);
            f/=numClasses;
        }
        return f;
    }
   
    protected double computeFScore(double[][] confMat, int c,double beta) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        //some very small non-zero value, in the extreme case that no cases of 
        //this class were correctly classified
        if (tp == .0)
            return .0000001; 
        
        double fp = 0.0, fn = 0.0;
        
        for (int i = 0; i < confMat.length; i++) {
            if (i!=c) {
                fp += confMat[i][c];
                fn += confMat[c][i];
            }
        }
        double precision = tp / (tp+fp);
        double recall = tp / (tp+fn);
        return (1+beta*beta) * (precision*recall) / ((beta*beta)*precision + recall);
    }
    protected double computeAROC(int c){
        class Pair implements Comparable<Pair>{
            Double d1;
            Double d2;
            public Pair(Double a, Double b){
                d1=a;
                d2=b;
            }
            @Override
            public int compareTo(Pair p) {
                return d1.compareTo(p.d1);
            }
        }
        
        ArrayList<Pair> p=new ArrayList<>();
        for(int i=0;i<numInstances;i++){
            Pair temp=new Pair(predictedClassProbabilities.get(i)[c],actualClassValues.get(i));
            p.add(temp);
        }
        Collections.sort(p);
// Sum up the FP rate
        
        return 0;
    } 
    
   public void writeAllStats(String str){
       OutFile out=new OutFile(str);
       
   }
   
   boolean hasInstanceData(){ return numInstances()!=0;}
   
    public static void main(String[] args) throws FileNotFoundException {
        String path="C:\\Users\\ajb\\Dropbox\\Results\\UCIResults\\RotFCV\\Predictions\\semeion\\testFold0.csv";
        ClassifierResults cr= new ClassifierResults();
        cr.loadFromFile(path);
        System.out.println("FILE TEST =\n"+cr.writeInstancePredictions());
    }
}
