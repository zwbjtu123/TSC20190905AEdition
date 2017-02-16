/*
This classifier is enhanced so that classifier builds a random forest with the 
facility to build by forward selection addition of trees to minimize OOB error.    

Further enhanced to include OOB error estimates and predictions

Further changes: 
1. set number of trees (m_numTrees) via grid search on a range (using OOB) that
defaults to 
{10 [Weka Default],100,200,.., 500 [R default],...,1000} (11 values)
2. set number of features  (max value m==numAttributes without class)
per tree (m_numFeatures) and m_numTrees through grid
search on a range 
1, 10, sqrt(m) [R default], log_2(m)+1 [Weka default], m [full set]}
(4 values)+add an option to choose randomly for each tree?
grid search is then just 55 values and because it uses OOB no CV is required
 */
package weka.classifiers.trees;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author ajb
 */
public class TunedRandomForest extends RandomForest implements SaveCVAccuracy{
    boolean tune=true;
    boolean tuneFeatures=true;
    boolean debug=false;
    int[] numTreesRange;
    int[] numFeaturesRange;
    double trainAcc;
    String trainPath="";
    Random rng;
    ArrayList<Double> accuracy;
    boolean crossValidate=true;
    boolean findTrainAcc=true;  //If there is no tuning, this will find the estimate with the fixed values
    
    public void setCrossValidate(boolean b){
        crossValidate=b;
    }
    public void setTrainAcc(boolean b){
        findTrainAcc=b;
    }

    
    public TunedRandomForest(){
        super();
        m_numTrees=500;
        m_numExecutionSlots=1; 
        m_bagger=new EnhancedBagging();
        rng=new Random();
        accuracy=new ArrayList<>();
    }
    public void setSeed(int s){
        super.setSeed(s);
        rng=new Random();
        rng.setSeed(s);
    }
    
    public void debug(boolean b){
        this.debug=b;
    }
    
    public void tuneTree(boolean b){
        tune=b;
    }
    public void tuneFeatures(boolean b){
        tuneFeatures=b;
    }
    public void setNumTreesRange(int[] d){
        numTreesRange=d;
    }
    public void setNumFeaturesRange(int[] d){
        numFeaturesRange=d;
    }
    @Override
    public void setCVPath(String train) {
        trainPath=train;
    }

    @Override
    public String getParameters() {
        String result="TrainAcc,"+trainAcc+",numTrees,"+this.getNumTrees()+",NumFeatures,"+this.getNumFeatures();
        for(double d:accuracy)
            result+=","+d;
        return result;
    }
    protected final void setDefaultGridSearchRange(int m){
//This only involves 55 or 44 parameter searches, unlike RBF that uses 625 by default.   
        if(debug)
            System.out.println("Setting defaults ....");
        numTreesRange=new int[11];
        numTreesRange[0]=10; //Weka default
        for(int i=1;i<numTreesRange.length;i++)
            numTreesRange[i]=100*i;
        if(tuneFeatures){
            if(m>10)//Include defaults for Weka (Utils.log2(m)+1) and R version  (int)Math.sqrt(m)
                numFeaturesRange=new int[]{1,10,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
            else
                numFeaturesRange=new int[]{1,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
        }
        else
            numFeaturesRange=new int[]{(int)Math.sqrt(m)};
            
  }

    protected final void setDefaultFeatureRange(int m){
//This only involves 55 or 44 parameter searches, unlike RBF that uses 625 by default.   
        if(debug)
            System.out.println("Setting default features....");

        if(tuneFeatures){
            if(m>10)//Include defaults for Weka (Utils.log2(m)+1) and R version  (int)Math.sqrt(m)
                numFeaturesRange=new int[]{1,10,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
            else
                numFeaturesRange=new int[]{1,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
        }
        else
            numFeaturesRange=new int[]{(int)Math.sqrt(m)};
            
  }
    
    
    double[][] OOBPredictions;
/*This 
    */    
    @Override
    public void buildClassifier(Instances data) throws Exception{
        int folds=10;
        if(crossValidate){
            if(folds>data.numInstances())
                folds=data.numInstances();
            if(debug)
                System.out.print(" Folds ="+folds);
        }
        
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        super.setSeed(rng.nextInt());
        if(tune){
            if(numTreesRange==null)
                setDefaultGridSearchRange(data.numAttributes()-1);
            else if(numFeaturesRange==null)
                setDefaultFeatureRange(data.numAttributes()-1);
            double bestErr=1.0;
  //Its a local nested class! urggg            
            class Pair{
                int x,y;
                Pair(int a, int b){
                    x=a;
                    y=b;
                }
            }
            ArrayList<Pair> ties=new ArrayList<>();
            for(int numFeatures:numFeaturesRange){//Need to start from scratch for each
                if(debug)
                    System.out.print(" numFeatures ="+numFeatures);
                
                for(int numTrees:numTreesRange){//Need to start from scratch for each                   t= new RandomForest();
                    RandomForest t= new RandomForest();
                    t.setSeed(rng.nextInt());
                    t.setNumFeatures(numFeatures);
                    t.setNumTrees(numTrees);
                    Instances temp=new Instances(data);
                    Evaluation eval=new Evaluation(temp);
                    double e;
                    if(crossValidate){  
                        eval.crossValidateModel(t, temp, folds, rng);
                        e=eval.errorRate();
                    }
                    else{
                        t.buildClassifier(temp);
                        e=t.measureOutOfBagError();
                    }
//                    double e=1-ClassifierTools.stratifiedCrossValidation(data,t, folds,0);
                    accuracy.add(1-e);
                    if(debug)
                        System.out.println(" numTrees ="+numTrees+" Acc = "+(1-e));
                    
//                    double e=t.measureOutOfBagError();
//                    System.out.println(" CV Error ="+e);
//                    t.addTrees(numTreesRange[j]-numTreesRange[j-1], data);
//                    double e=t.findOOBError();
                    if(e<bestErr){
                        bestErr=e;
                       ties=new ArrayList<>();//Remove previous ties
                        ties.add(new Pair(numFeatures,numTrees));
                    }
                    else if(e==bestErr){//tied best, chosen randomeli
                        ties.add(new Pair(numFeatures,numTrees));
   
                    }
                }
            }
            int bestNumTrees;
            int bestNumAtts;
            Pair best=ties.get(rng.nextInt(ties.size()));
            bestNumAtts=best.x;
            bestNumTrees=best.y;
            
            this.setNumTrees(bestNumTrees);
            this.setNumFeatures(bestNumAtts);
            trainAcc=1-bestErr;
            if(debug)
                System.out.println("Best num atts ="+bestNumAtts+" best num trees = "+bestNumTrees+" best train acc = "+trainAcc);
            if(trainPath!=""){  //Save train results not implemented
            }
        }
        else
            setNumTrees(Math.max(1,(int)Math.sqrt(data.numAttributes()-1)));
        
        super.buildClassifier(data);
        if(findTrainAcc){   //Need find train acc, either through CV or OOB
            if(crossValidate){  
                    RandomForest t= new RandomForest();
                    t.setNumFeatures(this.getNumFeatures());
                    t.setNumTrees(this.getNumTrees());
                    t.setSeed(rng.nextInt());
                    Instances temp=new Instances(data);
                    Evaluation eval=new Evaluation(temp);
                    double e;
                eval.crossValidateModel(t, data, folds, rng);
                trainAcc=1-eval.errorRate();
            }
            else{
                trainAcc=1-this.measureOutOfBagError();
            }
        }
        
        
/*
// Cant do this         super.buildClassifier(data);
//cos it recreates the bagger
    m_bagger = new EnhancedBagging();
    RandomTree rTree = new RandomTree();

        // set up the random tree options
        m_KValue = m_numFeatures;
        if (m_KValue < 1) m_KValue = (int) Utils.log2(data.numAttributes())+1;
        rTree.setKValue(m_KValue);
        rTree.setMaxDepth(getMaxDepth());

        // set up the bagger and build the forest
        m_bagger.setClassifier(rTree);
        m_bagger.setSeed(m_randomSeed);
        m_bagger.setNumIterations(m_numTrees);
        m_bagger.setCalcOutOfBag(true);
        m_bagger.setNumExecutionSlots(m_numExecutionSlots);
        m_bagger.buildClassifier(data);        
*/        
    }

    public void addTrees(int n, Instances data) throws Exception{
        EnhancedBagging newTrees =new EnhancedBagging();
        RandomTree rTree = new RandomTree();
        // set up the random tree options
        m_KValue = m_numFeatures;
        rTree.setKValue(m_KValue);
        rTree.setMaxDepth(getMaxDepth());
//Change this so that it is reproducable
        Random r= new Random();
        newTrees.setSeed(r.nextInt());
        newTrees.setClassifier(rTree);
        newTrees.setNumIterations(n);
        newTrees.setCalcOutOfBag(true);
        newTrees.setNumExecutionSlots(m_numExecutionSlots);
        newTrees.buildClassifier(data);
        newTrees.findOOBProbabilities();
//Merge with previous
        m_bagger.aggregate(newTrees);
        m_bagger.finalizeAggregation();
//Update OOB Error, as this is seemingly not done in the bagger
        
        m_numTrees+=n;
        m_bagger.setNumIterations(m_numTrees); 
        ((EnhancedBagging)m_bagger).mergeBaggers(newTrees);
        
    }
    public double getBaggingPercent(){
      return m_bagger.getBagSizePercent();
    }

    private class EnhancedBagging extends Bagging{
// 
        @Override
        public void buildClassifier(Instances data)throws Exception {
            super.buildClassifier(data);
            m_data=data;
//            System.out.println(" RESET BAGGER");

        }
        double[][] OOBProbabilities;
        int[] counts;
        public void mergeBaggers(EnhancedBagging other){
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_data.numClasses(); j++) {
                      OOBProbabilities[i][j]=counts[i]*OOBProbabilities[i][j]+other.counts[i]*other.OOBProbabilities[i][j];
                      OOBProbabilities[i][j]/=counts[i]+other.counts[i];
                }
                counts[i]=counts[i]+other.counts[i];
            }
//Merge  m_inBags index i is classifier, j the instance
            boolean[][] inBags = new boolean[m_inBag.length+other.m_inBag.length][];
            for(int i=0;i<m_inBag.length;i++)
                inBags[i]=m_inBag[i];
            for(int i=0;i<other.m_inBag.length;i++)
                inBags[m_inBag.length+i]=other.m_inBag[i];
            m_inBag=inBags;
            findOOBError();
        }
        public void findOOBProbabilities() throws Exception{
            OOBProbabilities=new double[m_data.numInstances()][m_data.numClasses()];
            counts=new int[m_data.numInstances()];
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_Classifiers.length; j++) {
                    if (m_inBag[j][i])
                      continue;
                    counts[i]++;
                    double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
                // average the probability estimates
                    for (int k = 0; k < m_data.numClasses(); k++) {
                        OOBProbabilities[i][k] += newProbs[k];
                    }
                }
                for (int k = 0; k < m_data.numClasses(); k++) {
                    OOBProbabilities[i][k] /= counts[i];
                }
            }
        }
        
        public double findOOBError(){
            double correct = 0.0;
            for (int i = 0; i < m_data.numInstances(); i++) {
                double[] probs = OOBProbabilities[i];
                int vote =0;
                for (int j = 1; j < probs.length; j++) {
                  if(probs[vote]<probs[j])
                      vote=j;
            }
            if(m_data.instance(i).classValue()==vote) 
                correct++;
            }
            m_OutOfBagError=1- correct/(double)m_data.numInstances();
//            System.out.println(" NEW OOB ERROR ="+m_OutOfBagError);
            return m_OutOfBagError;
        }
        
 //       public double getOOBError
    }
    public double findOOBError() throws Exception{
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        return ((EnhancedBagging)m_bagger).findOOBError();
    }
    public double[][] findOOBProbabilities() throws Exception{
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
    public double[][] getOBProbabilities() throws Exception{
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
  
  
  
    public static void main(String[] args) {
        
  //      testBinMaker();
  //      System.exit(0);
        DecimalFormat df = new DecimalFormat("##.###");
        try{
                String s="SwedishLeaf";
                System.out.println(" PROBLEM ="+s);
                Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TRAIN");
                Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TEST");
                TunedRandomForest rf=new TunedRandomForest();
               rf.buildClassifier(train);
                System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
                for(int i=0;i<5;i++){
                    System.out.println(" Number f trees ="+rf.getNumTrees()+" num elements ="+rf.numElements());
                    System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
                    double[][] probs=rf.findOOBProbabilities();
/*s
                    for (int j = 0; j < probs.length; j++) {
                        double[] prob = probs[j];
                        for (int k = 0; k < prob.length; k++) {
                            System.out.print(","+prob[k]);
                        }
                        System.out.println("");
                        
                    }
*/
                    rf.addTrees(50, train);
                }
                int correct=0;
                for(Instance ins:test){
                    double[] pred=rf.distributionForInstance(ins);
                    double cls=rf.classifyInstance(ins);
                    if(cls==ins.classValue())
                        correct++;
                }
                System.out.println(" ACC = "+((double)correct)/test.numInstances());
//                System.out.println(" calc out of bag? ="+rf.m_bagger.m_CalcOutOfBag);
                System.exit(0);
                double a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
                System.out.println(" error ="+df.format(1-a));
//                tsbf.buildClassifier(train);
 //               double c=tsbf.classifyInstance(test.instance(0));
 //               System.out.println(" Class ="+c);
        }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }
  
}
