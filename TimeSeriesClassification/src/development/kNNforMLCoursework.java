package development;

import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.neighboursearch.NearestNeighbourSearch;

/** Nearest neighbour classifier that extends the weka one but can take 
 * alternative distance functions.
 * @author ajb
 * @version 1.0
 * @since 5/4/09

1. Normalisation: set by method normalise(boolean)
2. Cross Validation: set by method crossValidate(int maxK)
3. Use weighting: set by the method weightVotes()

* 
 * */

public class kNNforMLCoursework extends IBk {
	protected DistanceFunction dist;
	double[][] distMatrix;
	boolean storeDistance;
	public kNNforMLCoursework(){
//Defaults to Euclidean distance		
            super();
            super.setKNN(1);
            setDistanceFunction(new EuclideanDistance());
            m_CrossValidate=false;
            m_kNNUpper=1;
            normalise(false);
	}
	public kNNforMLCoursework(int k){
            super(k);
            setDistanceFunction(new EuclideanDistance());
            m_CrossValidate=false;
            normalise(false);
	}
	public kNNforMLCoursework(DistanceFunction df){
            super();
            setDistanceFunction(df);
            m_CrossValidate=false;
            normalise(false);
	}
	
	public final void setDistanceFunction(DistanceFunction df){
		dist=df;
		NearestNeighbourSearch s = super.getNearestNeighbourSearchAlgorithm();
		try{
			s.setDistanceFunction(df);
		}catch(Exception e){
			System.err.println(" Exception thrown setting distance function ="+e+" in "+this);
                        e.printStackTrace();
                        System.exit(0);
		}
	}
	public double distance(Instance first, Instance second) {  
		  return dist.distance(first, second);
	  }

/** Set whether to normalise attributes to the range [0..1]. Returns true if 
 * successful   
     * @param v normalise or not  
     * @return  true if able to turn normalise on or off*/   
	public final boolean normalise(boolean v){
		if(dist instanceof NormalizableDistance){
			((NormalizableDistance)dist).setDontNormalize(!v);
                        return true;
                }
                return false;
	}
/** Set whether to select k by cross validation.
     * @param maxK maximum value to cross validate to*/
	public void useCrossValidation(int maxK){
            m_CrossValidate=true;
            m_kNNUpper=maxK;
            m_kNNValid=false;
	}
/** Set whether to use distance weighting. */
	public void distanceVote(){
            m_DistanceWeighting=WEIGHT_INVERSE;
	}
/** 
 * simply loads the file on path or exits the program
 * @param fullPath source path for ARFF file WITHOUT THE EXTENSION for some reason
 * @return Instances from path
 */
	public static Instances loadData(String fullPath)
	{
		Instances d=null;
		FileReader r;
		int nosAtts;
		try{		
			r= new FileReader(fullPath+".arff"); 
			d = new Instances(r); 
			d.setClassIndex(d.numAttributes()-1);
		}
		catch(Exception e)
		{
			System.out.println("Unable to load data on path "+fullPath+" Exception thrown ="+e);
			System.exit(0);
		}
		return d;
	}

/**
 * 	Simple util to find the accuracy of a trained classifier on a test set. Probably is a built in method for this! 
 * @param test
 * @param c
 * @return accuracy of classifier c on Instances test
 */
	public static double accuracy(Instances test, Classifier c){
		double a=0;
		int size=test.numInstances();
		Instance d;
		double predictedClass,trueClass;
		for(int i=0;i<size;i++)
		{
			d=test.instance(i);
			try{
				predictedClass=c.classifyInstance(d);
				trueClass=d.classValue();
				if(trueClass==predictedClass)
					a++;
//				System.out.println("True = "+trueClass+" Predicted = "+predictedClass);
			}catch(Exception e){
                            System.out.println(" Error with instance "+i+" with Classifier "+c.getClass().getName()+" Exception ="+e);
                            e.printStackTrace();
                            System.exit(0);
                        }
		}
		return a/size;
	}	
	
        static String[] getProblemNames(String root){
            File file = new File(root);
            String[] directories = file.list(new FilenameFilter() {
                @Override
                public boolean accept(File current, String name) {
                  return new File(current, name).isDirectory();
                }
              });
           System.out.println(Arrays.toString(directories));
           return directories;
            
            
        }
        public static void simpleTestSplitExample(){
            DecimalFormat df= new DecimalFormat("###.###");
            String path="C:\\Users\\ajb\\Dropbox\\UCI Classification Problems\\";
            String[] problems=getProblemNames(path);
            for(String fileName:problems){
                System.out.print("Problem ="+fileName+"\t,");

                Instances train=loadData(path+fileName+"\\"+fileName+"-train");
                Instances test=loadData(path+fileName+"\\"+fileName+"-test");
                try {
    // Normalisation comparison for 1-NN: normalises onthe range [0..1]
                    kNNforMLCoursework knn1=new kNNforMLCoursework();
                    kNNforMLCoursework knn2=new kNNforMLCoursework();
                    knn1.setKNN(1);
                    knn2.setKNN(1);
                    knn1=new kNNforMLCoursework();
                    knn2=new kNNforMLCoursework();
                    knn1.buildClassifier(train);
                    knn2.buildClassifier(train);
                    knn1.normalise(false);
                    knn2.normalise(true);
                    double a1=accuracy(test,knn1);
                    double a2=accuracy(test,knn2);
                    System.out.print(df.format(a1)+"\t,"+df.format(a2));
    //CV comparison:  1-NN vs setting k through cross validation

                    knn1=new kNNforMLCoursework();
                    knn2=new kNNforMLCoursework();
                    knn1.setKNN(1);
                    knn2.useCrossValidation(100);
                    knn1.buildClassifier(train);
                    knn2.buildClassifier(train);
                    a1=accuracy(test,knn1);
                     a2=accuracy(test,knn2);
                    System.out.print("\t,"+df.format(a1)+"\t,"+df.format(a2));
    // Distance weighting: 5-NN voting vs 5-NN distance weighting

                    knn1=new kNNforMLCoursework();
                    knn2=new kNNforMLCoursework();
                    knn1.normalise(false);
                    knn1.setKNN(5);
                    knn2.normalise(false);
                    knn1.setKNN(5);
                    knn2.distanceVote();
                    knn1.buildClassifier(train);
                    knn2.buildClassifier(train);
                    a1=accuracy(test,knn1);
                     a2=accuracy(test,knn2);
                    System.out.println("\t,"+df.format(a1)+"\t,"+df.format(a2));
                } catch (Exception ex) {
                    Logger.getLogger(kNNforMLCoursework.class.getName()).log(Level.SEVERE, null, ex);
                }
            } 
        }
        
        public static void main(String[] args){
            simpleTestSplitExample();
	}
}
