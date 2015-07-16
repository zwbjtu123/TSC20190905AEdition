package weka.classifiers;

import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.lazy.kNN;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.BagOfPatterns;
import weka.filters.timeseries.SAX;
import weka.filters.unsupervised.instance.Randomize;

/**
 * Converts instances into Bag Of Patterns form, then applies 1NN 
 * 
 * @author James
 */
public class SAXVSM implements Classifier {

    double[][] classWeights; //class weight matrix
    public kNN knn;
    
    private BagOfPatterns bop;
    private final int PAA_intervalsPerWindow;
    private final int SAX_alphabetSize;
    private final int windowSize;
    
    private final FastVector alphabet;
    
    public SAXVSM(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
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
         
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("SAXVSM_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        int numClasses = data.numClasses();
        int numInstances = data.numInstances();
        int numTerms = bop.dictionaryIndices.size();
        
        //store class values
        double[] classValues = data.attributeToDoubleArray(data.classIndex());
        
        //convert each series to bop form
        double[][] bopmatrix = new double[numInstances][];
        for (int i = 0; i < numInstances; i++)
            bopmatrix[i] = bop.buildBag(data.get(i));
                
        //initialise class weights
        classWeights = new double[numClasses][];
        for (int i = 0; i < numClasses; ++i) {
            classWeights[i] = new double[numTerms];
            for (int j = 0; j < numTerms; ++j)
                classWeights[i][j] = 0; //cant remember whether java default inits doubles to 0, meh
        }

        //build class bags
        for (int i = 0; i < numInstances; ++i)
            for (int j = 0; j < numTerms; ++j)
                classWeights[(int)classValues[i]][j] += bopmatrix[i][j];
        
        System.out.println("\n\npre weighting");
        for (int i = 0; i < numClasses; ++i) {
            for (int j = 0; j < numTerms; ++j)
                System.out.print(classWeights[i][j] +" ");
            System.out.println("");
        }
                
        //apply tf x idf
        for (int i = 0; i < numTerms; ++i) { //for each term
            double df = 0; //document frequency
            //as i read it 'number of bags where term t appears' 
            //NOT how many times it appears in total, e.g appears 3 times in bag a, 2 times in bag b, df = 2, NOT 5
            
            for (int j = 0; j < numClasses; ++j) //find how many classes this term appears in
                if (classWeights[j][i] != 0)
                    ++df;
            
            if (df != 0) { //if it appears
                if (df != numClasses) { //but not in all, apply weighting
                    for (int j = 0; j < numClasses; ++j) 
                        if (classWeights[j][i] != 0) 
                            classWeights[j][i] = Math.log(1 + classWeights[j][i]) * Math.log(numClasses / df);                
                }
                else { //appears in all
                    //check this just in case, done to avoid log calculations
                    //if df == num classes -> idf = log(N/df) = log(1) = 0 surely
                    //but not mentioned in paper, only df == 0 case mentioned, to avoid divide by 0
                    for (int j = 0; j < numClasses; ++j) 
                        classWeights[j][i] = 0;
                }      
            }
                
        }
        
        System.out.println("\n\npost weighting");
        for (int i = 0; i < numClasses; ++i) {
            for (int j = 0; j < numTerms; ++j)
                System.out.print(classWeights[i][j] +" ");
            System.out.println("");
        }
        
        //this code tried using instance.mergeinstance, KEEPING EVERYTHING SPARSE
        //'merge' actually just adds on the values of the second onto the first
        //i.e a = [1,2,3], b = [2,3,4], a.merge(b) = [1,2,3,2,3,4]
//        Instances classMatrix = new Instances(matrix, matrix.numClasses());
//        
//        //merge each series in each CLASS
//        double[] classValues = matrix.attributeToDoubleArray(matrix.classIndex());
//        
//        for (int c = 0; c < matrix.numClasses(); ++c) {
//            boolean firstInClass = true;
//            Instance firstClassInst = null;
//            
//            for (int i = 0; i < classValues.length; ++i) {
//                if (classValues[i] == c)
//                    if (firstInClass) {
//                        firstClassInst = matrix.get(i);
//                        firstInClass = false;
//                    }
//                    else
//                        firstClassInst = firstClassInst.mergeInstance(matrix.get(i));
//            }
//            
//            if (firstClassInst != null)
//                classMatrix.add(firstClassInst);
//            
//        }
//        
//        System.out.println("initialmatrix");
//        System.out.println("numClasses: " + matrix.numClasses());
//        System.out.println("numInsts: " + matrix.numInstances());
//        System.out.println("\nnewmatrix");
//        System.out.println("numClasses: " + classMatrix.numClasses());
//        System.out.println("numInsts: " + classMatrix.numInstances());
//        
//        System.out.println("\n\n" + classMatrix);
    }

    /**
     * Takes two vectors of equal length, and computes the cosine similarity between them.
     * 
     * i.e
     * cosine(theta) = A . B / ||A|| ||B||
     * 
     * @param a
     * @param b
     * @return 
     */
    public double cosineSimilarity(double[] a, double[] b) throws Exception {
        if (a.length != b.length)
            throw new Exception("Cannot calculate cosine similarity between vectors of different lengths "
                    + "(" + a.length + ", " + b.length + ")");
        
        double dotProd = 0.0, aMag = 0.0, bMag = 0.0;
        
        for (int i = 0; i < a.length; ++i) {
            dotProd += a[i]*b[i];
            aMag += a[i]*a[i];
            bMag += b[i]*b[i];
        }
        
        if (aMag == 0 || bMag == 0)
            return 0;
        
        aMag = Math.sqrt(aMag);
        bMag = Math.sqrt(bMag);
        
        return dotProd / (aMag * bMag);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        
        int numClasses = classWeights.length;
        
        double[] distribution = distributionForInstance(instance);
        
        double maxIndex = 0, max = distribution[0];
        for (int i = 1; i < numClasses; ++i)
            if (distribution[i] > max) {
                max = distribution[i];
                maxIndex = i;
            }
        
        return maxIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int numClasses = classWeights.length;
        
        double[] termFreqs = bop.buildBag(instance);
        double[] similarities = new double[numClasses];
        
        System.out.println("SIMILARITIES");
        double sum = 0.0;
        for (int i = 0; i < numClasses; ++i) {
            similarities[i] = cosineSimilarity(classWeights[i], termFreqs);
            sum+=similarities[i];
            System.out.println(similarities[i]);
        }
        System.out.println("");
        
        if (sum != 0) {
            //should normalise to range -1,1, scale/translate to 0,1 
            //again check this lazy stuff works
            NormalizeCase.standardNorm(similarities);
            for (int i = 0; i < numClasses; ++i) {
                similarities[i] *= 0.5;
                similarities[i] += 1;
            }
        }
        
        return similarities;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        System.out.println("SAXVSMtest\n\n");
        
        try {
            
            
            //very small dataset for testing by eye
//            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\Sheet2_Train2.arff");
//            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TwoClassV1.arff");
            Instances all = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\FiveClassV1.arff");
            all.deleteAttributeAt(0); //just name of bottle        
            
            Randomize rand = new Randomize();
            rand.setInputFormat(all);
            for (int i = 0; i < all.numInstances(); ++i) {
                rand.input(all.get(i));
            }
            rand.batchFinished();
            
            int trainNum = (int) (all.numInstances() * 0.7);
            int testNum = all.numInstances() - trainNum;
            
            Instances train = new Instances(all, trainNum);
            for (int i = 0; i < trainNum; ++i) 
                train.add(rand.output());
            
            Instances test = new Instances(all, testNum);
            for (int i = 0; i < testNum; ++i) 
                test.add(rand.output());
            
            System.out.println(all.numInstances());
            System.out.println(train.numInstances());
            System.out.println(test.numInstances());
            
//            Instances train = new Instances(all, 0, trainNum);
//            Instances test = new Instances(all, trainNum, testNum);
            
            SAXVSM vsm = new SAXVSM(6,3,100);
            vsm.buildClassifier(train);
            
//            System.out.println("");
//            for (int i = 0; i < test.numInstances(); i++) {
//                System.out.println("bleh" + vsm.classifyInstance(test.get(i)));
//            }
            
            System.out.println("\nACCURACY TEST");
            System.out.println(ClassifierTools.accuracy(test, vsm));

            
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        
    }
}
