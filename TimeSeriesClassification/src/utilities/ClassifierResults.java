
package utilities;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;

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
    public double nll; //the true-class only version
    public double nllFull; //the (original) full-distribution version 
    public double meanAUROC;
    public double stddev; //across cv folds
    
    public double[][] confusionMatrix; //[actual class][predicted class]
    public double[] countPerClass;
//Used to avoid infinite NLL scores when prob of true class =0 or 
//prob of wrong class =1    
    public static double NLL_PENALTY=-20.0;
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
    
    /**
     * for building results one by one while testing, call finaliseResults
 to populate the ClassifierResults object once testing is finished
     */
    public ClassifierResults(int numClasses) {
        actualClassValues= new ArrayList<>();
        predictedClassValues = new ArrayList<>();
        predictedClassProbabilities = new ArrayList<>();
        
        this.numClasses = numClasses;
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

    public int getNumClasses() {
        return numClasses;
    }

    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    public int getNumInstances() {
        return numInstances;
    }

    public void setNumInstances(int numInstances) {
        this.numInstances = numInstances;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getParas() {
        return paras;
    }

    public void setParas(String paras) {
        this.paras = paras;
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
   
   public String writeResultsFileToString() throws IOException {                
        StringBuilder st = new StringBuilder();
        st.append(name).append("\n");
        st.append("BuildTime,").append(buildTime).append(",").append(paras).append("\n");
        st.append(acc).append("\n");
        
        if (predictedClassValues != null) {
            for(int i = 0; i < predictedClassValues.size();i++) {
                st.append(actualClassValues.get(i)).append(",").append(predictedClassValues.get(i)).append(","); //pred
                
                double[] distForInst=getDistributionForInstance(i);
                if (distForInst != null)
                    for (int j = 0; j < distForInst.length; j++)
                        st.append("," + distForInst[j]);

                st.append("\n");
            }
        }
        
        return st.toString();
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
       nllFull=findNLLFullDistribution();
       meanAUROC=findMeanAUROC();
           
    }
   
    /**
     * uses all probabilities in the distribution
     */
    public double findNLLFullDistribution(){
        double nll=0;
        for(int i=0;i<actualClassValues.size();i++){
            double[] dist=getDistributionForInstance(i);
            double trueClass=actualClassValues.get(i);
//                System.out.println(" instance "+i+" class = "+trueClass+" prob true class = "+dist[(int)trueClass]);
            double temp=0;
            for(int j=0;j<dist.length;j++){
                if(j!=(int)trueClass){
                    if(dist[j]==1)
                        temp+=NLL_PENALTY;
                    else{
                        temp+=Math.log(1-dist[j])/Math.log(2);

                    }
                }
                else{
                    if(dist[j]==0)
                        temp+=NLL_PENALTY;
                    else{
                        temp+=Math.log(dist[j])/Math.log(2);//Log 2
                    }
                }
            }
//               System.out.println(" Instance  "+i+" has NLL ="+temp);
            nll+=temp;
        }
        return -nll/actualClassValues.size();
    }
    
    /**
     * uses only the probability of the true class
     */
    public double findNLL(){
        double nll=0;
        for(int i=0;i<actualClassValues.size();i++){
            double[] dist=getDistributionForInstance(i);
            int trueClass = actualClassValues.get(i).intValue();
            
            if(dist[trueClass]==0)
                nll+=NLL_PENALTY;
            else
                nll+=Math.log(dist[trueClass])/Math.log(2);//Log 2
        }
        return -nll/actualClassValues.size();
    }
           
    public double findMeanAUROC(){
        double a=0;
        if(numClasses==2){
                a=findAUROC(1);
/*            if(countPerClass[0]<countPerClass[1])
            else
                a=findAUROC(1);
 */       }
        else{
            for(int i=0;i<numClasses;i++){
                a+=findAUROC(i);
            }
            a/=numClasses;
        }
        return a;
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
       b/=cm.length;
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
                f=findConfusionMatrixStats(cm,0,1);
            else
                f=findConfusionMatrixStats(cm,1,1);
        }
        else{//Average over all of them
            for(int i=0;i<numClasses;i++)
                f+=findConfusionMatrixStats(cm,i,1);
            f/=numClasses;
        }
        return f;
    }
   
    protected double findConfusionMatrixStats(double[][] confMat, int c,double beta) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        //some very small non-zero value, in the extreme case that no cases of 
        //this class were correctly classified
        if (tp == .0)
            return .0000001; 
        
        double fp = 0.0, fn = 0.0,tn=0.0;
        
        for (int i = 0; i < confMat.length; i++) {
            if (i!=c) {
                fp += confMat[i][c];
                fn += confMat[c][i];
                tn+=confMat[i][i];
            }
        }
        precision = tp / (tp+fp);
        recall = tp / (tp+fn);
        sensitivity=recall;
        specificity=tn/(fp+tn);
        return (1+beta*beta) * (precision*recall) / ((beta*beta)*precision + recall);
    }
    protected double findAUROC(int c){
        class Pair implements Comparable<Pair>{
            Double x;
            Double y;
            public Pair(Double a, Double b){
                x=a;
                y=b;
            }
            @Override
            public int compareTo(Pair p) {
                return p.x.compareTo(x);
            }
            public String toString(){ return "("+x+","+y+")";}
        }
        
        ArrayList<Pair> p=new ArrayList<>();
        double nosPositive=0,nosNegative;
        for(int i=0;i<numInstances;i++){
            Pair temp=new Pair(predictedClassProbabilities.get(i)[c],actualClassValues.get(i));
            if(c==actualClassValues.get(i))
                nosPositive++;
            p.add(temp);
        }
        nosNegative=actualClassValues.size()-nosPositive;
        Collections.sort(p);
//        System.out.println(" List = "+p.toString());
/* http://www.cs.waikato.ac.nz/~remco/roc.pdf
        Determine points on ROC curve as follows; 
        starts in the origin and goes one unit up, for every
negative outcome the curve goes one unit to the right. Units on the x-axis
are 1
#TN and on the y-axis 1
#TP where #TP (#TN) is the total number
of true positives (true negatives). This gives the points on the ROC curve
(0; 0); (x1; y1); : : : ; (xn; yn); (1; 1).
        */
        ArrayList<Pair> roc=new ArrayList<>();
        double x=0;
        double oldX=0;
        double y=0;
        int xAdd=0, yAdd=0;
        boolean xLast=false,yLast=false;
        roc.add(new Pair(x,y));
        for(int i=0;i<numInstances;i++){
            if(p.get(i).y==c){
                if(yLast)
                    roc.add(new Pair(x,y));
                xLast=true;
                yLast=false;
                x+=1/nosPositive;
                xAdd++;
                if(xAdd==nosPositive)
                    x=1.0;
                
            }
            else{ 
                if(xLast)
                    roc.add(new Pair(x,y));
                yLast=true;
                xLast=false;
                y+=1/nosNegative;
                yAdd++;
                if(yAdd==nosNegative)
                    y=1.0;
            }
        }
        roc.add(new Pair(1.0,1.0));
//        System.out.println(" ROC  = "+roc.toString());
/* Calculate the area under the ROC curve, as the sum over all trapezoids with
base xi+1 to xi , that is, A
*/
//        System.out.println("Number of points ="+roc.size());    
        double auroc=0;
        for(int i=0;i<roc.size()-1;i++){
            auroc+=(roc.get(i+1).y-roc.get(i).y)*(roc.get(i+1).x);
        }
        return auroc;
    } 
    
   public String writeAllStats(){
        String str="Acc,"+acc+"\n";
        str+="BalancedAcc,"+balancedAcc+"\n"; 
        str+="sensitivity,"+sensitivity+"\n"; 
        str+="precision,"+precision+"\n"; 
        str+="recall,"+recall+"\n"; 
        str+="specificity,"+specificity+"\n";         
        str+="f1,"+f1+"\n"; 
        str+="nll,"+nll+"\n"; 
        str+="nllFull,"+nllFull+"\n";
        str+="meanAUROC,"+meanAUROC+"\n"; 
        str+="stddev,"+stddev+"\n"; 
        str+="Count per class:\n";
        for(int i=0;i<countPerClass.length;i++)
            str+="Class "+i+","+countPerClass[i]+"\n";
        str+="Confusion Matrix:\n";
        for(int i=0;i<confusionMatrix.length;i++){
            for(int j=0;j<confusionMatrix[i].length;j++)
                str+=confusionMatrix[i][j]+",";
            str+="\n";
        }
        return str;
   }
   
   boolean hasInstanceData(){ return numInstances()!=0;}

    public static void main(String[] args) throws FileNotFoundException {
        
        String path="C:\\JamesLPHD\\testFold1.csv";
//        String path="C:\\JamesLPHD\\testFold0.csv";
//        String path="C:/JamesLPHD/TwoClass.csv";
        ClassifierResults cr= new ClassifierResults();
        cr.loadFromFile(path);
        cr.findAllStats();
        System.out.println("AUROC = "+cr.meanAUROC);
        System.out.println("FILE TEST =\n"+cr.writeAllStats());
        OutFile out=new OutFile("C:\\JamesLPHD\\testFold1stats.csv");
        out.writeLine(cr.writeAllStats());
    }
}
