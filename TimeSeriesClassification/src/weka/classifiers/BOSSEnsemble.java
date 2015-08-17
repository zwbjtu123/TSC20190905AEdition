//TODO
//What strucutre to store BOSSWindows in
//Currently, jsut storing in list
//If current windowsize classifier (C) passes threshold test (T(C)) for the CURRENT max
//      put it in the list
//Later as it progresses the 'current max' will obviously keep increasing though
//And C's that once passed no longer do
//Think of a data structure to keep all the elements ordered (by accuracy), allow fast insertion/deletion
//and fast search for windows no longer within the threshold 
//because we generally want to reduce space used


package weka.classifiers;

import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import utilities.ClassifierTools;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import development.DataSets;
import static JamesStuff.TestTools.testClassifiers;

/**
 * BOSS classifier with parameter search, if parameters are known, use BOSS classifier and directly provide them.
 * 
 * Params: normalise? (i.e should first fourier coefficient(mean value) be discarded)
 * Alphabetsize fixed to four, as in BOSS paper
 * 
 * @author James
 */
public class BOSSEnsemble implements Classifier {
    
    public static boolean debug = true;
    
    public List<BOSSWindow> classifiers;
    
    private final double minWindowFactor = 1.0/10.0;
    private final double maxWindowFactor = 10.0/10.0;
    private final double correctThreshold = 0.92;
    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int alphabetSize = 4;
    private boolean norm;
    
    public BOSSEnsemble(boolean normalise) {
        norm = normalise;
    }

    private static class BOSSWindow implements Comparable<BOSSWindow> { 
        private BOSS classifier;
        public final double accuracy;

        public BOSSWindow(BOSS classifer, double accuracy) {
            this.classifier = classifer;
            this.accuracy = accuracy;
        }

        public double classifyInstance(Instance inst) throws Exception { 
            return classifier.classifyInstance(inst); 
        }
        
        public int getWindowSize()   { return classifier.getWindowSize();  }
        public int getWordLength()   { return classifier.getWordLength();  }
        public int getAlphabetSize() { return classifier.getAlphabetSize(); }
        public boolean isNorm()      { return classifier.isNorm(); }
        
        @Override
        public int compareTo(BOSSWindow other) {
            if (this.accuracy > other.accuracy) 
                return 1;
            if (this.accuracy == other.accuracy) 
                return 0;
            return -1;
        }
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        //todo validate and all taht
        
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSSEnsemble_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        classifiers = new LinkedList<>();
        
        int seriesLength = data.numAttributes()-1;
        int minWindow = (int)(seriesLength * minWindowFactor);
        int maxWindow = (int)(seriesLength * maxWindowFactor);     
    
        //keep track of current max accuracy, constantly check for correctthreshold to discard to save space
        double maxAcc = -1.0;
        
        for (int winSize = minWindow; winSize < maxWindow; ++winSize) {          
            BOSS boss = new BOSS(wordLengths[0], alphabetSize, winSize, norm);  
            boss.buildClassifier(data); //initial setup, with max word length     
            
            
            BOSS bestClassifierForWinSize = null; 
            double bestWindowAcc = -1.0;
            
            for (Integer wordLen : wordLengths) {            
                boss = BOSS.shortenHistograms(wordLen, boss); //in first case, same lengths (16 by default, max length), will do nothing
                
                assert(boss.bags.size() == data.numInstances()); //sanity check
                
                int correct = 0, numSeries = data.numInstances(); 
                for (int i = 0; i < numSeries; ++i) {
                    double c = boss.classifyInstance(i); //classify series i, ignoring histogram i
                    if (c == data.get(i).classValue())
                        ++correct;
                }
                
                double acc = (double)correct/(double)numSeries;

                if (acc > bestWindowAcc) {
                    bestWindowAcc = acc;
                    bestClassifierForWinSize = boss;
                }
            }
            
            //if not within correct threshold of the CURRENT max, dont bother storing at all
            //will still likely be some by the end of this build that dont fall within threshold
            //because classifiers at the start may have passed a lower threshold than the later ones 
            if (bestWindowAcc > maxAcc * correctThreshold) {
                classifiers.add(new BOSSWindow(bestClassifierForWinSize, bestWindowAcc));

                if (bestWindowAcc > maxAcc)
                    maxAcc = bestWindowAcc;       
            }
        }
        
        Iterator<BOSSWindow> it = classifiers.iterator();
        
        while (it.hasNext()) {
            BOSSWindow boss = it.next();
            if (boss.accuracy < maxAcc * correctThreshold)
                it.remove();
        } //get rid of any extras that dont fall within the final max threshold
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        
        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) 
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
        
        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double mostAcc = Collections.max(classifiers).accuracy;
        
        double[] classHist = new double[instance.numClasses()];
        double sum = 0;
        for (BOSSWindow classifier : classifiers) {
                double classification = classifier.classifyInstance(instance);
                classHist[(int)classification]++;
                sum++;
        }
        
        if (sum != 0)
            for (int i = 0; i < classHist.length; ++i)
                classHist[i] /= sum;
        
        return classHist;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args){
        System.out.println("BOSSEnsembleTest\n\n");
        
        fullTest();
        
//        try {
//            Instances train = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\Car\\\\Car_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\\\Temp\\\\TESTDATA\\\\TSC Problems\\\\Car\\\\Car_TEST.arff");
//            
//            long start = System.currentTimeMillis();
//            
//            BOSSEnsemble boss = new BOSSEnsemble(true);
//            boss.buildClassifier(train);
//            System.out.println("Training Complete: " + (System.currentTimeMillis() - start) + "ms");
//            
//            start = System.currentTimeMillis();
//            System.out.println("ACC: " + ClassifierTools.accuracy(test, boss));
//            System.out.println("Accuracy test time: "+(System.currentTimeMillis() - start) + "ms");
//        }
//        catch (Exception e) {
//            System.out.println(e);
//            e.printStackTrace();
//        }
        
    }
    
    public static void fullTest() {
        System.out.println("BOSSEnsembleFullTest");
        try {
            String[] datasets = DataSets.ucrNames;
            BOSSEnsemble boss = new BOSSEnsemble(true);
            
            Classifier[] classifier = { boss } ;
            String[] cNames = { "BOSSEnsemble" };
            
            testClassifiers(classifier, cNames, datasets, "BOSSEnsembleUCRTest.csv", 0);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
    
    public static void debug(String msg) {
        if (debug) 
            System.out.println(msg);
    }
}