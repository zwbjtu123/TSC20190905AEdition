package development.Jay;

import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.DTW;
import weka.core.elastic_distance_measures.DTW_DistanceBasic;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class ResultsChecker {

    public static final String dir = "C:/users/ajb/Dropbox/TSC Problems/";
    
    public static void main(String[] args) throws Exception{
        
        
        String dataName = "ItalyPowerDemand";
        System.out.print(dataName+",");
        for(int seed = 1; seed <= 100; seed++){
            System.out.print(trainTestDTW(dataName, seed)+",");
        }
        
    }
        
    public static double trainTestDTW(String dataName, int seed) throws Exception{
        
        Instances train = ClassifierTools.loadData(dir+dataName+"/"+dataName+"_TRAIN");
        Instances test = ClassifierTools.loadData(dir+dataName+"/"+dataName+"_TEST");
        Instances[] seeded = InstanceTools.resampleTrainAndTestInstances(train, test, seed);
        
        train = seeded[0];
        test = seeded[1];
        
        kNN knn = new kNN();
//        BasicDTW dtw = new BasicDTW();
        DTW_DistanceBasic dtw = new DTW_DistanceBasic();
        knn.setDistanceFunction(dtw);
        DTW_1NN knn2=new DTW_1NN();
        knn2.setR(1.0);
        knn2.optimiseWindow(false);
        
        kNN knn3 = new kNN();
        knn3.setDistanceFunction(new DTW());

        knn.buildClassifier(train);
        knn2.buildClassifier(train);
        knn3.buildClassifier(train);
        int correct = 0;
        int c2 = 0;
        int c3 = 0;
        
        for(int i = 0; i < test.numInstances(); i++){
            if(knn.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
            if(knn2.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                c2++;
            }
            if(knn3.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                c3++;
            }
        }
        System.out.println(" knn1 ="+(double)correct/test.numInstances()+" knn2 ="+(double)c2/test.numInstances()+" knn3 = "+(double)c3/test.numInstances());
        return (double)correct/test.numInstances();

    }    
    
    
    
    
}
