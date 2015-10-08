package JamesStuff;

import fileIO.OutFile;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Various methods related to testing classifiers 
 * 
 * @author James
 */
public class TestTools {
    
    /**
     * Returns train [0] and test [1] Instances for the given dataset
     * 
     * @param name name of TSC Problems dataset
     * @return { trainset, testset }
     * @throws Exception 
     */
    public static Instances[] loadTestTrainsets(String path, String name) throws Exception {
        Instances train = ClassifierTools.loadData(path+name+"\\"+name+"_TRAIN");
        Instances test = ClassifierTools.loadData(path+name+"\\"+name+"_TEST");
        
        return new Instances[] { train, test };
    }
    
    /**
     * 
     * @param classifier to test, ready to go (i.e in a state where buildClassifier can be called)
     * @param data = { trainset, testset }
     * @param resamples if 0, will just test accuracy on train/test sets as they come (deterministic), if > 0, will repeat tests, 
     *                     randomly resampling test/train sets (while maintaining class distribution)
     * @return accuracy of classifier on given dataset
     * @throws Exception 
     */
    public static double testAccuracy(Classifier classifier, Instances[] data, int resamples) throws Exception {
        if (resamples == 0) { 
            classifier.buildClassifier(data[0]);
            return ClassifierTools.accuracy(data[1], classifier);
        }
        
        //else resampling to be done
        double thisAcc = 0, totalAcc = 0;
        for (int testno = 0; testno < resamples; testno++) {
            data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], testno);

            classifier.buildClassifier(data[0]);

            thisAcc = ClassifierTools.accuracy(data[1], classifier);
            totalAcc += thisAcc;
        }
        return totalAcc / resamples;
    }

    /**
     * Performs ClassifierTools.accuracy tests on each classifier for each dataset, and outputs
     * error rates for each to the provided output CSV file.
     * 
     * todo: any error/validity checking on datasets etc if wanted
     * add dataset path parameter instead of hard coding it
     * 
     * @param classifiers array of ready-to-use classifiers (i.e any needed parameters are set)
     * @param classifierNames parallel array of names/labels for each classifier
     * @param datasetNames names(NOT full paths) of datasets to be loaded from TSCProblems datasets, e.g ("GunPoint")
     * @param fullOutputFile full path directory and name of the CSV file to write results to, e.g ("C:\\path\\file.csv")
     * @param resamples if 0, will just test accuracy on train/test sets as they come (deterministic), if > 0, will repeat tests, 
     *                     randomly resampling test/train sets (while maintaining class distribution)
     * @throws Exception
     */
    public static void testClassifiers(Classifier[] classifiers, String[] classifierNames, String[] datasetNames, String fullOutputFile, int resamples) 
            throws Exception {
        
        String arffPath="C:\\Temp\\TESTDATA\\TSC Problems\\";
        
        OutFile out = new OutFile(fullOutputFile);
        
        //header info
        out.writeString("Dataset, NumClasses, TrainsetSize, TestsetSize, SeriesLength");
        for (int i = 0; i < classifierNames.length; ++i)
            out.writeString(", " + classifierNames[i]);
        out.newLine();
        
        for (int i = 0; i < datasetNames.length; ++i) {
            System.out.println(i + ": " + datasetNames[i]);
            
            //[0] = train, [1] = test
            Instances[] datasets = loadTestTrainsets(arffPath, datasetNames[i]);

            //dataset info
            out.writeString(datasetNames[i] + ", " + 
                    datasets[0].numClasses() + ", " + 
                    datasets[0].numInstances() + ", " + 
                    datasets[1].numInstances() + ", " + 
                    (datasets[0].numAttributes()-1));

            //class accuracies on this dataset
            for (int j = 0; j < classifiers.length; ++j)
                out.writeString(", " + (1.0 - testAccuracy(classifiers[j], datasets, resamples))); // 1- to get error rate
            out.newLine();
            
        } 
    }
    
}
