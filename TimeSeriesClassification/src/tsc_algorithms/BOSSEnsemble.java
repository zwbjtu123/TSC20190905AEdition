package tsc_algorithms;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import utilities.ClassifierTools;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

/**
 * BOSS classifier with parameter search, if parameters are known, use 'BOSS' classifier and directly provide them.
 * 
 * Params: normalise? (i.e should first fourier coefficient(mean value) be discarded)
 * Alphabetsize fixed to four
 * 
 * @author James
 */
public class BOSSEnsemble implements Classifier {

    private static boolean debug = false;
    
    private List<BOSSWindow> classifiers;

    private final double correctThreshold = 0.92;
    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int alphabetSize = 4;
    private boolean norm;
     
    public BOSSEnsemble(boolean normalise) {
        norm = normalise;
    }

    public static class BOSSWindow implements Comparable<BOSSWindow> { 
        private BOSS classifier;
        public final double accuracy;

        public BOSSWindow(BOSS classifer, double accuracy) {
            this.classifier = classifer;
            this.accuracy = accuracy;
        }

        public double classifyInstance(Instance inst) throws Exception { 
            return classifier.classifyInstance(inst); 
        }
        
        /**
         * @return { numIntervals(word length), alphabetSize, slidingWindowSize } 
         */
        public int[] getParameters() { return classifier.getParameters();  }
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
    
     /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize } for each BOSSWindow in this *built* classifier
     */
    public int[][] getParameters() {
        int[][] params = new int[classifiers.size()][];
        int i = 0;
        for (BOSSWindow boss : classifiers) 
            params[i++] = boss.getParameters();
         
        return params;
    }
    
    @Override
    public void buildClassifier(final Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSSEnsemble_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        classifiers = new LinkedList<BOSSWindow>();
        
        int numSeries = data.numInstances();
        
        int seriesLength = data.numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = seriesLength; 

        //int winInc = 1; //check every window size in range
        
        //whats the max number of window sizes that should be searched through
        //double maxWindowSearches = 200; 
        double maxWindowSearches = Math.min(200, Math.sqrt(seriesLength)); 
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches); 
        if (winInc < 1) winInc = 1;
        
        
        //keep track of current max window size accuracy, constantly check for correctthreshold to discard to save space
        double maxAcc = -1.0;
        
        for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {          
            BOSS boss = new BOSS(wordLengths[0], alphabetSize, winSize, norm);  
            boss.buildClassifier(data); //initial setup for this windowsize, with max word length     
            
            BOSS bestClassifierForWinSize = null; 
            double bestAccForWinSize = -1.0;

            //find best word length for this window size
            for (Integer wordLen : wordLengths) {            
                boss = BOSS.shortenHistograms(wordLen, boss); //in first iteration, same lengths (wordLengths[0] == max length), will do nothing

                int correct = 0; 
                for (int i = 0; i < numSeries; ++i) {
                    double c = boss.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                    if (c == data.get(i).classValue())
                        ++correct;
                }
                
                double acc = (double)correct/(double)numSeries;     
                if (acc > bestAccForWinSize) {
                    bestAccForWinSize = acc;
                    bestClassifierForWinSize = boss;
                }
            }
 
            //if not within correct threshold of the CURRENT max, dont bother storing at all
            //will still likely be some by the end of this build that dont fall within threshold
            //because classifiers at the start may have passed a lower threshold than the later ones 
            if (bestAccForWinSize > maxAcc * correctThreshold) {
                classifiers.add(new BOSSWindow(bestClassifierForWinSize, bestAccForWinSize));

                if (bestAccForWinSize > maxAcc)
                    maxAcc = bestAccForWinSize;       
            }
        }
        
         //get rid of any extras that dont fall within the final max threshold
        Iterator<BOSSWindow> it = classifiers.iterator();
        while (it.hasNext()) {
            BOSSWindow boss = it.next();
            if (boss.accuracy < maxAcc * correctThreshold)
                it.remove();
        }
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
        double[] classHist = new double[instance.numClasses()];
        
        //get votes from all windows 
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
        basicTest();
    }
    
    public static void basicTest() {
        System.out.println("BOSSEnsembleBasicTest\n");
        try {
            debug = true;
            
            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");
            
            System.out.println(train.relationName());
            
            BOSSEnsemble boss = new BOSSEnsemble(true);
            System.out.println("Training starting");
            long start = System.nanoTime();
            boss.buildClassifier(train);
            double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Training done (" + trainTime + "s)");
            
            System.out.println("Ensemble Size: " + boss.classifiers.size());
            System.out.println("Param sets: ");
            int[][] params = boss.getParameters();
            for (int i = 0; i < params.length; ++i)
                System.out.println(i + ": " + params[i][0] + " " + params[i][1] + " " + params[i][2] + " " + boss.norm);
            
            System.out.println("\nTesting starting");
            start = System.nanoTime();
            double acc = ClassifierTools.accuracy(test, boss);
            double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Testing done (" + testTime + "s)");
            
            System.out.println("\nACC: " + acc);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
}