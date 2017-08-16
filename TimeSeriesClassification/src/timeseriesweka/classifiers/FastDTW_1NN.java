package timeseriesweka.classifiers;
import timeseriesweka.elastic_distance_measures.DTW;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
import java.util.HashMap;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.*;

/* DO NOT USE: Not properly debugged. This class is a specialisation of kNN that
can only be used with the efficient DTW distance
 * 
 * The reason for specialising is this class has the option of searching for 
the optimal window length
 * through a grid search of values.
 * 
 * By default this class does not do a search of window size.  
 * To search for the window size call
 * optimiseWindow(true);
 * By default, this does a leave one out cross validation on every possible 
window size, then sets the proportion to the one with the largest accuracy,
ties taking the smallest window (slow)
. This will be slow, and not how the Keogh group do it. They do a stepwise increase
of window by 1% until there is no improvement for three steps. 

This has two possible speedups

1. Optimize window. This starts at full window, w=100%, and records the maximum warp 
 made over the data set, say k. Rather than move to w=w-1 it moves to w=k if k<w-1,
thus saving many evaluations

2. Early abandon on a window. If, during the accuracy calculation for a single window size,
the accuracy cannot be better than the best so far, we can quit. 

3. Early abandon on the nearest neighbour calculation. One obvious speed up is 
to store the distance matrix for a given window size. This requires O(n^2) extra
memory and means you cannot early abandon individual distances. 

The problem with this is it means you cannot 


CHECK THIS: For implementation reasons, a window size of 1 
is equivalent to Euclidean distance (rather than a window size of 0
 */

public class FastDTW_1NN implements Classifier {
    private boolean optimiseWindow=false;
    private double windowSize=1;
    private int maxPercentageWarp=100;
    private Instances train;
    private int trainSize;
    private int bestWarp;
    private int maxWindowSize;
    DTW_DistanceBasic dtw;
    HashMap<Integer,Double> distances;
    double maxR=1;

    public FastDTW_1NN(){
        dtw=new DTW();
    }
    public FastDTW_1NN(DTW_DistanceBasic d){
        dtw=d;
    }
    
    
    public double getMaxR(){ return maxR;}
    public void setMaxPercentageWarp(int a){maxPercentageWarp=a;}
    public void optimiseWindow(boolean b){ optimiseWindow=b;}
    public void setR(double r){dtw.setR(r);}
    public double getR(){ return dtw.getR();}
    public int getBestWarp(){ return bestWarp;}
    public int getWindowSize(){ return dtw.getWindowSize(train.numAttributes()-1);}

    @Override
    public void buildClassifier(Instances d){
        train=d;
        trainSize=d.numInstances();
        if(optimiseWindow){
            maxR=0;
            double maxAcc=0;
            int dataLength=train.numAttributes()-1;
/*  If the data length < 100 then there will be some repetition
            should skip some values I reckon
            if(dataLength<maxNosWindows)
                maxPercentageWarp=dataLength;
        */

            for(int i=maxPercentageWarp;i>=0;i-=1){
        //Set r for current value as the precentage of series length.
//                dtw=new DTW();
               
                dtw.setR(i/100.0);

/*Can do an early abandon inside cross validate. If it cannot be more accurate 
 than maxR even with some left to evaluate then stop evaluation
*/                
                double acc=crossValidateAccuracy(maxAcc);
                if(acc>=maxAcc){
                    maxR=i;
                    maxAcc=acc;
               }
//                System.out.println(" r="+i+" warpsize ="+x+" train acc= "+acc+" best acc ="+maxR);
/* Can ignore all window sizes bigger than the max used on the previous iteration
*/                
                
               if(maxWindowSize<(i-1)*dataLength/100){
                   System.out.println("WINDOW SIZE ="+dtw.getWindowSize()+" Can reset downwards at "+i+"% to ="+((int)(100*(maxWindowSize/(double)dataLength))));
                   i=(int)(100*(maxWindowSize/(double)dataLength));
                   i++;
//                   i=Math.round(100*(maxWindowSize/(double)dataLength))/100;
               } 

            }
            bestWarp=(int)(maxR*dataLength/100);
            System.out.println("OPTIMAL WINDOW ="+maxR+" % which gives a warp of"+bestWarp+" data");
  //          dtw=new DTW();
            dtw.setR(maxR/100.0);
        }
    }
    @Override
    public double classifyInstance(Instance d){
/*Basic distance, with early abandon. This is only for 1-nearest neighbour*/
            double minSoFar=Double.MAX_VALUE;
            double dist; int index=0;
            for(int i=0;i<train.numInstances();i++){
                    dist=dtw.distance(train.instance(i),d,minSoFar);
                    if(dist<minSoFar){
                            minSoFar=dist;
                            index=i;
                    }
            }
            return train.instance(index).classValue();
    }
    @Override
    public double[] distributionForInstance(Instance instance){
        double[] dist=new double[instance.numClasses()];
        dist[(int)classifyInstance(instance)]=1;
        return dist;
    }

    
    /**Could do this by calculating the distance matrix, but then 	
 * you cannot use the early abandon. Early abandon about doubles the speed,
 * as will storing the distances. Given the extra n^2 memory, probably better
 * to just use the early abandon. We could store those that were not abandoned?
answer is to store those without the abandon in a hash table indexed by i and j,
*where index i,j == j,i

* @return 
 */
    private  double crossValidateAccuracy(double maxAcc){
        double a=0,d, minDist;
        int nearest;
        Instance inst;
        int bestNosCorrect=(int)(maxAcc*trainSize);
        maxWindowSize=0;
        int w;
        distances=new HashMap<>(trainSize);
        
        
        for(int i=0;i<trainSize;i++){
//Find nearest to element i
            nearest=0;
            minDist=Double.MAX_VALUE;
            inst=train.instance(i);
            for(int j=0;j<trainSize;j++){
                if(i!=j){
//  d=dtw.distance(inst,train.instance(j),minDist);
//Store past distances if not early abandoned 
//Not seen i,j before                    
                  if(j>i){
                        d=dtw.distance(inst,train.instance(j),minDist);
                        //Store if not early abandon
                        if(d!=Double.MAX_VALUE){
//                            System.out.println(" Storing distance "+i+" "+j+" d="+d+" with key "+(i*trainSize+j));
                            distances.put(i*trainSize+j,d);
//                            storeCount++;
                        }
//Else if stored recover                        
                    }else if(distances.containsKey(j*trainSize+i)){
                        d=distances.get(j*trainSize+i);
//                       System.out.println(" Recovering distance "+i+" "+j+" d="+d);
//                        recoverCount++;
                    }
//Else recalculate with new early abandon                    
                    else{
                        d=dtw.distance(inst,train.instance(j),minDist);
                    }        
                    if(d<minDist){
                        nearest=j;
                        minDist=d;
                        w=dtw.findMaxWindow();
                        if(w>maxWindowSize)
                            maxWindowSize=w;
                    }
                }
            }
                //Measure accuracy for nearest to element i			
            if(inst.classValue()==train.instance(nearest).classValue())
                a++;
           //Early abandon if it cannot be better than the best so far. 
            if(a+trainSize-i<bestNosCorrect){
//                    System.out.println(" Early abandon on CV when a="+a+" and i ="+i+" best nos correct = "+bestNosCorrect+" maxAcc ="+maxAcc+" train set size ="+trainSize);
                return 0.0;
            }
        }
//        System.out.println("trainSize ="+trainSize+" stored ="+storeCount+" recovered "+recoverCount);
        return a/(double)trainSize;
    }
    public static void main(String[] args){
            FastDTW_1NN c = new FastDTW_1NN();
            String path="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\";

            Instances test=ClassifierTools.loadData(path+"Coffee\\Coffee_TEST.arff");
            Instances train=ClassifierTools.loadData(path+"Coffee\\Coffee_TRAIN.arff");
            train.setClassIndex(train.numAttributes()-1);
            c.buildClassifier(train);

    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
