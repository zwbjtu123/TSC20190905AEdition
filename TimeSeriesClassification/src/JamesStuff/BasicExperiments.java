//package JamesStuff;
//
//import development.DataSets;
//import fileIO.OutFile;
//import utilities.ClassifierTools;
//import utilities.InstanceTools;
//import weka.classifiers.Classifier;
//import tsc_algorithms.BagOfPatterns;
//import tsc_algorithms.SAXVSM;
//import tsc_algorithms.SAX_1NN;
//import weka.classifiers.lazy.DTW_1NN;
//import weka.classifiers.lazy.kNN;
//import weka.core.Instances;
//import weka.filters.unsupervised.instance.Randomize;
//
///**
// * Mostly just a playground, can't remember if there's a currently useful output,
// * but probably useful code in places
// * 
// * Originally used to compare SAXy classifiers on the UCR datasets, but TestTools has the more proper methods for this now
// * 
// * @author James
// */
//public class BasicExperiments {
//
//    private static final int SAXalphaSize = 4;
//    private static final int PAAinterSize = 6;
//    private static final int windowSize = 100;
//    
//    public static String[] UCRnames = {
//        "SyntheticControl",
//        "GunPoint",
//        "CBF",
//        "FaceAll",
//        "OSULeaf",
//        "SwedishLeaf",
//        "fiftywords",
//        "Trace",
//        "TwoPatterns",
//        "wafer",
//        "FaceFour",
//        "Lightning2",
//        "Lightning7",
//        "ECGFiveDays",
//        "Adiac",
//        "yoga",
//        "fish",
//        "Beef",
//        "Coffee",
//        "OliveOil"
//    };
//   
//    public static String[] UCRvsmdata = {
//        "Car",
//        "GunPoint",
//        "CBF",
//        "ChlorineConcentration",
//        "Coffee",
//        "FaceAll",
//        "FaceFour",
//        "fish",
//        "Haptics",
//        "InlineSkate",
//        "OliveOil",
//        "wafer"
//    };
//    
//    public static void main(String[] args) throws Exception{
//        
//        //CURRENTLY SKIPPING DTW
//        //annoyingly long time to run, can copy results from elsewhere anyways
//        
//        //UCRdata x 10 matrix
//        //columns = num classes, train size, test size, series length, 1nn acc, dtw acc, sax1nn acc, bop acc, saxvsm acc
//        double[][] tableData = new double[UCRnames.length][];
//        
//        String[] classifierNames = {
//            "1NN", "SAX_1NN", "SAXVSM", "BOP_1NN"
//        };
//        
//        String[] columnHeadings = {
//            "Dataset", "NumClasses", "TrainsetSize", "TestsetSize", "SeriesLength", 
//            classifierNames[0], classifierNames[1], classifierNames[2], classifierNames[3], 
//            "GIVENBOP_1NN"
//        };
//        
//        double[] givenBopErrRates = {
//            0.037,
//            0.027,
//            0.013,
//            0.219,
//            0.256,
//            0.198,
//            0.466,
//            0.0,
//            0.129,
//            0.003,
//            0.023,
//            0.164,
//            0.466,
//            0.150,
//            0.432,
//            0.170,
//            0.074,
//            0.433,
//            0.036,
//            0.133
//        };
//        
//        for (int i = 0; i < UCRnames.length; ++i) {
//            //[0] = train, [1] = test
//            Instances[] datasets = loadTestTrainsets(UCRnames[i]);
//            tableData[i] = getDataInfo(datasets);
//            
//            int[] params = BagOfPatterns.getUCRParameters(UCRnames[i]);
//            
//            kNN knn = new kNN();
//            //DTW_1NN dtw = new DTW_1NN();
//            SAX_1NN sax1nn = new SAX_1NN(params[1], params[2]);
//            SAXVSM saxvsm = new SAXVSM(params[1], params[2], params[0]);
//            BagOfPatterns bop = new BagOfPatterns(params[1], params[2], params[0]);
//            
//            
//            Classifier[] cs = { knn, sax1nn, saxvsm, bop };
//
//            for (int c = 0; c < cs.length; ++c)
//                tableData[i][c+4] = 1 - testAccuracy(cs[c], datasets, 0); // 1- to get error rate
//        }
//        
//        OutFile out = new OutFile("saxishClassifierTests.csv");
//        
//
//        for (String heading : columnHeadings)
//            out.writeString(heading + ",");
//        out.newLine();
//        
//        for (int i = 0; i < UCRnames.length; ++i) {
//            out.writeString(UCRnames[i] + ",");
//            for (int j = 0; j < tableData[i].length; ++j)
//                out.writeString(tableData[i][j] + ",");
//            out.writeDouble(givenBopErrRates[i]);
//            out.newLine();
//        }
//        
//        out.closeFile();
//    }
//    
//
//    
//    private static double[] getDataInfo(Instances[] data) {
//        double[] result = new double[8];
//        
//        result[0] = data[0].numClasses();
//        result[1] = data[0].numInstances();
//        result[2] = data[1].numInstances();
//        result[3] = data[0].numAttributes()-1;
//        //rest will be accuracies determined by testing 
//        return result;
//    }
//    
//    
//    
//    
//    
//    
//    
//    
//    /**
//     * Returns train [0] and test [1] Instances for the given dataset
//     * 
//     * @param name name of TSC Problems dataset
//     * @return { trainset, testset }
//     * @throws Exception 
//     */
//    public static Instances[] loadTestTrainsets(String name) throws Exception {
//        String arffPath="C:\\Temp\\TESTDATA\\TSC Problems\\";
//        
//        Instances train = ClassifierTools.loadData(arffPath+name+"\\"+name+"_TRAIN");
//        Instances test = ClassifierTools.loadData(arffPath+name+"\\"+name+"_TEST");
//        
//        return new Instances[] { train, test };
//    }
//    
//    /**
//     * 
//     * @param classifier to test, ready to go
//     * @param data = { trainset, testset }
//     * @param resamples if 0, will just test accuracy on train/test sets as they come (deterministic), if > 0, will repeat tests, 
//     *                     randomly resampling test/train sets (while maintaining class distribution)
//     * @return average(if resamples>1) accuracy of classifier on given dataset
//     * @throws Exception 
//     */
//    public static double testAccuracy(Classifier classifier, Instances[] data, int resamples) throws Exception {
//        if (resamples == 0) { 
//            classifier.buildClassifier(data[0]);
//            return ClassifierTools.accuracy(data[1], classifier);
//        }
//        
//        double thisAcc = 0, totalAcc = 0;
//        for (int testno = 0; testno < resamples; testno++) {
//            data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], testno);
//
//            classifier.buildClassifier(data[0]);
//
//            thisAcc = ClassifierTools.accuracy(data[1], classifier);
//            totalAcc += thisAcc;
//        }
//        return totalAcc / resamples;
//    }
//
//    /**
//     * Performs ClassifierTools.accuracy tests on each classifier for each dataset, and outputs
//     * error rates for each to the provided output CSV file
//     * 
//     * @param classifiers array of ready-to-use classifiers (i.e any needed parameters are set)
//     * @param classifierNames parallel array of names/labels for each classifier
//     * @param datasetNames names(NOT full paths) of datasets to be loaded from TSCProblems datasets, e.g ("GunPoint")
//     * @param fullOutputFile full path directory and name of the CSV file to write results to, e.g ("C:\\path\\file.csv")
//     * @param resamples if 0, will just test accuracy on train/test sets as they come (deterministic), if > 0, will repeat tests, 
//     *                     randomly resampling test/train sets (while maintaining class distribution)
//     * @throws Exception
//     */
//    public static void testClassifiers(Classifier[] classifiers, String[] classifierNames, String[] datasetNames, String fullOutputFile, int resamples) 
//            throws Exception {
//        OutFile out = new OutFile(fullOutputFile);
//        
//        //header info
//        out.writeString("Dataset, NumClasses, TrainsetSize, TestsetSize, SeriesLength");
//        for (int i = 0; i < classifierNames.length; ++i)
//            out.writeString(", " + classifierNames[i]);
//        out.newLine();
//        
//        for (int i = 0; i < datasetNames.length; ++i) {
//            //[0] = train, [1] = test
//            Instances[] datasets = loadTestTrainsets(datasetNames[i]);
//
//            //dataset info
//            out.writeString(datasetNames[i] + ", " + 
//                    datasets[0].numClasses() + ", " + 
//                    datasets[0].numInstances() + ", " + 
//                    datasets[1].numInstances() + ", " + 
//                    (datasets[0].numAttributes()-1));
//
//            //class accuracies on this dataset
//            for (int j = 0; j < classifiers.length; ++j)
//                out.writeString(", " + (1.0 - testAccuracy(classifiers[j], datasets, resamples))); // 1- to get error rate
//            out.newLine();
//            
//        } 
//    }
//    
//    
//}
