/*
 */
package weka.classifiers.meta;

import weka.classifiers.trees.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author ajb
 */
public class TunedRotationForest extends RotationForest implements SaveCVAccuracy{
    boolean tune=true;
    int[] numTreesRange;
    int[] numFeaturesRange;
    double trainAcc;
    String trainPath="";
    boolean tuneFeatures=false;
    boolean debug=false;
    boolean findTrainAcc=true;
    Random rng;
    ArrayList<Double> accuracy;
    
    public TunedRotationForest(){
        super();
        this.setNumIterations(200);
        rng=new Random();
        accuracy=new ArrayList<>();
        
    }   
    
    public void setSeed(int s){
        super.setSeed(s);
        rng=new Random();
        rng.setSeed(s);
    }
     public void tuneFeatures(boolean b){
        tuneFeatures=b;
    }
     public void debug(boolean b){
        this.debug=b;
    }
     public void estimateAccFromTrain(boolean b){
        this.findTrainAcc=b;
    }
  
    public void tuneTree(boolean b){
        tune=b;
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
        String result="TrainAcc,"+trainAcc+",numTrees,"+this.getNumIterations()+",NumFeatures,"+this.getMaxGroup();
        for(double d:accuracy)
            result+=","+d;
        
        return result;
    }
    protected final void setDefaultGridSearchRange(int m){
        numTreesRange=new int[11];
        numTreesRange[0]=10;
        for(int i=1;i<numTreesRange.length;i++)
            numTreesRange[i]=(50)*i;  //This may be slow!
        
        if(tuneFeatures){
            if(m>10)//Weka default is 3.
                numFeaturesRange=new int[]{3,10,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m/10};
            else
                numFeaturesRange=new int[]{3,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m};
        }
        else
            numFeaturesRange=new int[]{3};
            
    }

    protected final void setDefaultFeatureRange(int m){
//This only involves 55 or 44 parameter searches, unlike RBF that uses 625 by default.   
        if(debug)
            System.out.println("Setting default features....");

        if(tuneFeatures){
            if(m>10)//Include defaults for Weka (Utils.log2(m)+1) and R version  (int)Math.sqrt(m)
                numFeaturesRange=new int[]{3,10,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
            else
                numFeaturesRange=new int[]{3,(int)Math.sqrt(m),(int) Utils.log2(m)+1,m-1};
        }
        else
            numFeaturesRange=new int[]{3};
  }
    
    
    
    @Override
    public void buildClassifier(Instances data) throws Exception{
        int folds=10;
        if(folds>data.numInstances())
            folds=data.numInstances();
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
                    System.out.println(" numFeatures ="+numFeatures);
                for(int numTrees:numTreesRange){//Need to start from scratch for each
                
                    RotationForest t= new RotationForest();
                    t.setMaxGroup(numFeatures);
                    t.setMinGroup(numFeatures);
                    t.setNumIterations(numTrees);
                    Instances temp=new Instances(data);
                    Evaluation eval=new Evaluation(temp);
                    t.setSeed(rng.nextInt());
                    eval.crossValidateModel(t, temp, folds, rng);
                    double e=eval.errorRate();
                    if(debug)
                        System.out.println("\t numTrees ="+numTrees+" Acc = "+(1-e));
                    accuracy.add(1-e);
                    if(e<bestErr){
                        bestErr=e;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new Pair(numFeatures,numTrees));
                    }
                    else if(e==bestErr){//Sort out ties
                        ties.add(new Pair(numFeatures,numTrees));
                    }
                }
            }
            int bestNumTrees=0;
            int bestNumAtts=0;
 
            Pair best=ties.get(rng.nextInt(ties.size()));
            bestNumAtts=best.x;
            bestNumTrees=best.y;
            this.setNumIterations(bestNumTrees);
            this.setMaxGroup(bestNumAtts);
            this.setMinGroup(bestNumAtts);
            trainAcc=1-bestErr;
            if(debug)
                System.out.println("Best num atts ="+bestNumAtts+" best num trees="+bestNumTrees+" "+bestNumTrees+" best Acc ="+trainAcc);
            if(trainPath!=""){  //Save train results NOT IMPLEMENTED
                
            }
        }
/*If there is no parameter search, then there is no train CV available.        
this gives the option of finding one. It is inefficient
*/        
        else if(findTrainAcc){
             RotationForest t= new RotationForest();
            t.setMaxGroup(this.getMaxGroup());
            t.setMinGroup(this.getMinGroup());
            t.setNumIterations(this.getNumIterations());
            Instances temp=new Instances(data);
            Evaluation eval=new Evaluation(temp);
            t.setSeed(rng.nextInt());
            eval.crossValidateModel(t, temp, folds, rng);
            trainAcc=1-eval.errorRate();
        }
        super.buildClassifier(data);
    }
  
  
    public static void main(String[] args) {
    }
  
}
