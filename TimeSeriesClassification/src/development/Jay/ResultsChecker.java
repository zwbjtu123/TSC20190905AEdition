package development.Jay;

import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.DTW_DistanceBasic;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class ResultsChecker {

    public static final String dir = "C:/Temp/Dropbox/TSC Problems/";
    
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

        knn.buildClassifier(train);
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){
            if(knn.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
        }
        
        return (double)correct/test.numInstances();

    }    
    
    
    
    
}
