/*
Time Series Bag of Features (TSBF): Baydogan

Time series classification with a bag-of-features (TSBF) algorithm.
series length =m, num series =n
1. Subsequences are sampled and partitioned into intervals for feature extraction.

    #minimum interval length wmin=5;   
    number of subseries numSub= floor(m/wmin)-d
    each subseries is of random length ls
    each subseries is split into d segments
    mean, variance and slope is extracted for each segment

For i=1 to number of  subsequences
    select start and end point s1 and s2
    for each time series t in T
                    generate intervals on t_s1 and t_s2
                    generate features (mean, std dev and slope) from intervals
                    add to new features for t
 

	nos of features per sub series=3*d+4
        nos features per series = numSub*(3*d+4)

This forms a new data set that is identical to TSF except for the global features.

2. "Each subsequence feature set is labeled with the class of the time series and 
each time series forms the bag." 
I think it works by building a random forest on the labelled transformed subseries and use the 
class probability estimates from the forest. 


binsize=10      #bin size for codebook generation   

3. A classifier generates class probability estimates. 

4. Histograms of the class probability estimates are generated (and concatenated) to summarize the subsequence
information. 

5. Global features are added. 

6. A final classifier is then trained on the new representation to assign each time series.
 */
package tsc_algorithms;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class TSBF extends AbstractClassifier{
    int wmin=5;   
    int numSub;
    int numIntervals;
    int seriesLength;
    double[] zLevels={0.1,0.25,0.5,0.75}; //minimum subsequence length factors (z) to be evaluated
    double zLevel=zLevels[0];
    boolean paramSearch=true;
    int binsize=10;      //bin size for codebook generation   
    int[][] subSeries;

    Instances formatIntervalInstances(Instances data){
        int numFeatures=numSub*(3*numIntervals+4);
        //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
        for(int j=0;j<numFeatures;j++){
                name = "F"+j;
                atts.addElement(new Attribute(name));
        }
        //Get the class values as a fast vector			
        Attribute target =data.attribute(data.classIndex());
       FastVector vals=new FastVector(target.numValues());
        for(int j=0;j<target.numValues();j++)
                vals.addElement(target.value(j));
        atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
//create blank instances with the correct class value                
        Instances result = new Instances("Tree",atts,data.numInstances());
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            DenseInstance in=new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes()-1,data.instance(i).classValue());
            result.add(in);
        }
        return result;
        
    }
    @Override
    public void buildClassifier(Instances data) throws Exception {
        seriesLength=data.numAttributes()-1;
        numIntervals=(int)((zLevel*seriesLength)/wmin);
        numSub=  (seriesLength/wmin)-numIntervals;
        subSeries =new int[numSub][2];
 /*                   //build our test and train sets. for cross-validation.
                    for (int l = 0; l < noFolds; l++) {
                        Instances trainCV = data.trainCV(noFolds, l);
                        Instances testCV = data.testCV(noFolds, l);
*/

//Build first transform
        Instances features=formatIntervalInstances(data);
        for(int i=0;i<numSub;i++){
            //Generate subsequences
            
            //Generate intervals
            
            for(int j=0;j<data.numInstances();j++){
//Generate features for sequence i                
            }
            
            
        }
//    each subseries is of random length ls
//    each subseries is split into d segments
//    mean, variance and slope is extracted for each segment

//1. choose intervals        
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
