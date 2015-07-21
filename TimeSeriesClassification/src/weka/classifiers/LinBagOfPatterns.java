package weka.classifiers;

import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.timeseries.BagOfPatterns;
import weka.filters.timeseries.SAX;

/**
 * Converts instances into Bag Of Patterns form, then applies 1NN 
 * 
 * @author James
 */
public class LinBagOfPatterns implements Classifier {

    public Instances matrix;
    public kNN knn;
    
    private BagOfPatterns bop;
    private final int PAA_intervalsPerWindow;
    private final int SAX_alphabetSize;
    private final int windowSize;
    
    private final FastVector alphabet;
    
    public LinBagOfPatterns(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.PAA_intervalsPerWindow = PAA_intervalsPerWindow;
        this.SAX_alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        
        bop = new BagOfPatterns(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
        
        knn = new kNN(); //default to 1NN, Euclidean distance

        alphabet = SAX.getAlphabet(SAX_alphabetSize);
    }
    
    public int getPAA_intervalsPerWindow() {
        return PAA_intervalsPerWindow;
    }

    public int getSAX_alphabetSize() {
        return SAX_alphabetSize;
    }

    public int getWindowSize() {
        return windowSize;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        Instances dataCopy = new Instances(data);
        matrix = bop.process(data);
        
//        Instances localCopy = new Instances(data); 
//        
//        buildDictionary();
//        dictionaryAttributes.add(data.classAttribute());
//        
//        matrix = new Instances("BagOfPatterns", dictionaryAttributes, data.numInstances());
//        matrix.setClassIndex(matrix.numAttributes()-1);
//        
//        for (int i = 0; i < data.numInstances(); i++) {
//            double[] hist = buildBag(data.get(i));
//            
//            matrix.add(new SparseInstance(1.0, hist));
//            matrix.get(i).setClassValue(data.get(i).classValue()); 
//            //ask again about this, seems so dumb, deep copying data again jsut to set class value
//        }
        
        knn.buildClassifier(matrix);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //convert to proper form
        double[] hist = bop.bagToArray(bop.buildBag(instance));
        
        Instances newInsts = new Instances(matrix, 1); //copy attribute data
        newInsts.add(new SparseInstance(1.0, hist));
        //newInsts.firstInstance().setClassValue(instance.classValue());

        return knn.classifyInstance(newInsts.firstInstance());
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        //convert to proper form
        double[] hist = bop.bagToArray(bop.buildBag(instance));
        
        Instances newInsts = new Instances(matrix, 1); //copy attribute data
        newInsts.add(new SparseInstance(1.0, hist));
        //newInsts.firstInstance().setClassValue(instance.classValue());

        return knn.distributionForInstance(newInsts.firstInstance());
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        System.out.println("BagofPatternsTest\n\n");
        
        try {
            //very small dataset for testing by eye
//            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\Sheet2_Train2.arff");
            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TwoClassV1.arff");
            all.deleteAttributeAt(0); //just name of bottle
            
            
            int trainNum = (int) (all.numInstances() * 0.7);
            int testNum = all.numInstances() - trainNum;
            
            Instances train = new Instances(all, 0, trainNum);
            Instances test = new Instances(all, trainNum, testNum);
            
//            System.out.println("RAW TRAIN DATA");
//            System.out.println(train);
//            
//            System.out.println("\nRAW TEST DATA");
//            System.out.println(test);
            
            LinBagOfPatterns bop = new LinBagOfPatterns(6,3,100);
            bop.buildClassifier(train);
            
            System.out.println(bop.matrix);
 
//            System.out.println("\n\nDICTIONARY");
//            System.out.println(bop.dictionaryAttributes);
//            
//            System.out.println("\n\nTRAIN BOP");
//            System.out.println(bop.matrix);
//            
//            Instances testM = new Instances(bop.matrix, test.numInstances()); //copy attribute data
//            for (int i = 0; i < test.numInstances(); i++) {
//                double[] hist = bop.buildBag(test.get(i));
//                testM.add(new SparseInstance(1.0, hist));
//            }
//            
//            System.out.println("\n\nTEST BOP");
//            System.out.println(testM);
//            
//            System.out.println("");
//            for (int i = 0; i < test.numInstances(); i++) {
//                System.out.println(bop.classifyInstance(test.get(i)));
//            }
//            
//            System.out.println("");
//            for (int i = 0; i < test.numInstances(); ++i) {
//                double[] dist = bop.distributionForInstance(test.get(i));
//                
//                for (int j = 0; j < dist.length; ++j) 
//                    System.out.print(dist[j] + " ");
//                
//                System.out.println("");
//            }
            
            System.out.println("\nACCURACY TEST");
            System.out.println(ClassifierTools.accuracy(test, bop));

            
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        
    }
    
    @Override
    public String toString() { 
        return "BagOfPatterns";
    }
    
    /**
     * Returns a list of optimal parameters (winSize, intervals, alphabetSize) for a SUBSET OF the UCR datasets as given 
     * in the Lin paper, if dataset properties not found, returns default 50, 6, 4.
     * 
     * @param dataset 
     * @return array size 3 { winSize, intervals, alphabetSize }
     */
    public static int[] getUCRParameters(String dataset) {
        for (int i = 0; i < UCRnames.length; ++i)
            if (UCRnames[i].equals(dataset))
                return UCRparameters[i];
        
        return new int[] { 50, 6, 4 };
    }
    
    //parameters for winsize/intervals/alphabet for each ucr dataset copied from
    //'Rotation-invarient similarity in time series using bag-of-patterns representation' Lin etal 2012
    //[0] = windowsize
    //[1] = intervals 
    //[2] = alphabetsize
    public static int[][] UCRparameters = {
        { 24, 4, 3 },
        { 32, 4, 9 }, 
        { 32, 4, 4 },
        { 32, 8, 3 },
        { 64, 4, 4 },
        { 40, 8, 4 },
        { 80, 8, 3 },
        { 48, 4, 4 },
        { 32, 4, 4 },
        { 32, 8, 5 },
        { 64, 4, 4 },
        { 128, 8, 4 }, 
        { 64, 8, 4 },
        { 32, 8, 4 },
        { 32, 8, 9 },
        { 80, 4, 9 },
        { 128, 8, 7 },
        { 80, 4, 2 },
        { 48, 4, 3 },
        { 160, 4, 6 }
    };
    
    public static String[] UCRnames = {
        "SyntheticControl",
        "GunPoint",
        "CBF",
        "FaceAll",
        "OSULeaf",
        "SwedishLeaf",
        "fiftywords",
        "Trace",
        "TwoPatterns",
        "wafer",
        "FaceFour",
        "Lightning2",
        "Lightning7",
        "ECGFiveDays",
        "Adiac",
        "yoga",
        "fish",
        "Beef",
        "Coffee",
        "OliveOil"
    };
}
