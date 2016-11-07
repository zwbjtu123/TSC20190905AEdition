/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tsc_algorithms.boss;

import fileIO.OutFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import utilities.Timer;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * BoTSW classifier with parameter search and ensembling, if parameters are known, 
 * use 'BoTSW.java' classifier and directly provide them.
 * 
 * Will use BOSSDistance by default. If svm wanted, call setUseSVM(true). Precise SVM implementation/accuracy could not be recreated, likewise 
 * for kmeans, epsilon value ignored
 *  
 * @author James Large
 * 
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class BoTSWEnsemble implements Classifier, SaveCVAccuracy /*, HiveCoteModule*/ {
    
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Bailly, Adeline and Malinowski, Simon and Tavenard, Romain and Guyet, Thomas and Chapel, Laetitia");
        result.setValue(TechnicalInformation.Field.TITLE, "Bag-of-Temporal-SIFT-Words for Time Series Classification");
        result.setValue(TechnicalInformation.Field.JOURNAL, "ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");

        return result;
    }
    
    private List<BoTSWWindow> classifiers; 

    private final double correctThreshold = 0.92;
    private int maxEnsembleSize = Integer.MAX_VALUE;
    
    private final Integer[] n_bRanges = { 4, 8, 12, 16, 20 };
    private final Integer[] aRanges = { 4, 8 };
    private final Integer[] kRanges = { 32, 64, 128, 256, 512, 1024 };
    private final Integer[] csvmRanges = {1, 10, 100}; //not currently used, using BOSSDistance
    private final int alphabetSize = 4;
    //private boolean norm;
    
    public enum SerialiseOptions { 
        //dont do any seriealising, run as normal
        NONE, 
        
        //serialise the final botsw classifiers which made it into ensemble (does not serialise the entire BoTSWEnsemble object)
        //slight runtime cost 
        STORE, 
        
        //serialise the final botsw classifiers, and delete from main memory. reload each from ser file when needed in classification. 
        //the most memory used at any one time is therefore ~2 individual botsw classifiers during training. 
        //massive runtime cost, order of magnitude 
        STORE_LOAD 
    };
    
    
    private SerialiseOptions serOption = SerialiseOptions.NONE;
    private static String serFileLoc = "BOSSWindowSers\\";
     
    private boolean[] normOptions;
    
    private String trainCVPath;
    private boolean trainCV=false;

    private Instances train;
    private double ensembleCvAcc = -1;

    public static class BoTSWWindow implements Comparable<BoTSWWindow>, Serializable { 
        private BoTSW classifier;
        public double accuracy;
        public String filename;
        
        private static final long serialVersionUID = 2L;

        public BoTSWWindow(String filename) {
            this.filename = filename;
        }
        
        public BoTSWWindow(BoTSW classifer, double accuracy, String dataset) {
            this.classifier = classifer;
            this.accuracy = accuracy;
            buildFileName(dataset);
        }

        public double classifyInstance(Instance inst) throws Exception { 
            return classifier.classifyInstance(inst); 
        }
        
        public double classifyInstance(int test) throws Exception { 
            return classifier.classifyInstance(test); 
        }
        
        private void buildFileName(String dataset) {
            filename = serFileLoc + dataset + "_" + classifier.params.toString() + ".ser";
        }
        
        public boolean storeAndClearClassifier() {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
                out.writeObject(this);
                out.close();   
                clearClassifier();
                return true;
            }catch(IOException e) {
                System.out.print("Error serialiszing to " + filename);
                e.printStackTrace();
                return false;
            }
        }
        
        public boolean store() {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
                out.writeObject(this);
                out.close();         
                return true;
            }catch(IOException e) {
                System.out.print("Error serialiszing to " + filename);
                e.printStackTrace();
                return false;
            }
        }
        
        public void clearClassifier() {
            classifier = null;
        }
        
        public boolean load() {
            BoTSWWindow bw = null;
            try {
                ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
                bw = (BoTSWWindow) in.readObject();
                in.close();
                this.accuracy = bw.accuracy;
                this.classifier = bw.classifier;
                return true;
            }catch(IOException i) {
                System.out.print("Error deserialiszing from " + filename);
                i.printStackTrace();
                return false;
            }catch(ClassNotFoundException c) {
                System.out.println("BoTSWWindow class not found");
                c.printStackTrace();
                return false;
            }
        }
        
        public boolean deleteSerFile() {
            try {
                File f = new File(filename);
                return f.delete();
            } catch(SecurityException s) {
                System.out.println("Unable to delete, access denied: " + filename);
                s.printStackTrace();
                return false;
            }
        }
        
        /**
         * @return { numIntervals(word length), alphabetSize, slidingWindowSize } 
         */
        public String getParameters() { return classifier.getParameters();  }
        public int[] getParametersValues() { return classifier.getParametersValues();  }
        public int getNB() { return classifier.params.n_b;  }
        public int getA() { return classifier.params.a;  }
        public int getK() { return classifier.params.k;  }
        
        @Override
        public int compareTo(BoTSWWindow other) {
            if (this.accuracy > other.accuracy) 
                return 1;
            if (this.accuracy == other.accuracy) 
                return 0;
            return -1;
        }
    }
    
    @Override
    public void setCVPath(String train) {
        trainCVPath=train;
        trainCV=true;
    }

    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        
        BoTSWWindow first = classifiers.get(0);
        sb.append(first.getParameters());
            
        for (int i = 1; i < classifiers.size(); ++i) {
            BoTSWWindow botsw = classifiers.get(i);
            sb.append(",").append(botsw.getParameters());
        }
        
        return sb.toString();
    }
    
    @Override
    public int setNumberOfFolds(Instances data){
        return data.numInstances();
    }
    
     /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize } for each BoTSWWindow in this *built* classifier
     */
    public int[][] getParametersValues() {
        int[][] params = new int[classifiers.size()][];
        int i = 0;
        for (BoTSWWindow botsw : classifiers) 
            params[i++] = botsw.getParametersValues();
         
        return params;
    }
    
    public void setSerOption(SerialiseOptions option) { 
        serOption = option;
    }
    
    public void setSerFileLoc(String path) {
        serFileLoc = path;
    }
    
    public void setMaxEnsembleSize(int max) {
        maxEnsembleSize = max;
    }
    
    @Override
    public void buildClassifier(final Instances data) throws Exception {
        //Timer.PRINT = true; //timer will ignore print request by default, similar behaviour to NDEBUG
        
        this.train=data;
        
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSSEnsemble_BuildClassifier: Class attribute not set as last attribute in dataset");
 
        if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD) {
            DateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
            Date date = new Date();
            serFileLoc += data.relationName() + "_" + dateFormat.format(date) + "\\";
            File f = new File(serFileLoc);
            if (!f.isDirectory())
                f.mkdirs();
        }
        
        classifiers = new LinkedList<BoTSWWindow>();
        int numSeries = data.numInstances();
        //keep track of current max window size accuracy, constantly check for correctthreshold to discard to save space
        double maxAcc = -1.0;
        double minMaxAcc = -1.0; //the acc of the worst member to make it into the final ensemble as it stands
        
        boolean firstBuild = true;
        BoTSW.FeatureDiscoveryData[] fdData = null; //keypoint location and guassian of series data
        for (Integer n_b : n_bRanges) {
            Timer n_bTimer = new Timer("n_b="+n_b);
            
            for (Integer a : aRanges) {
                if (n_b*a > data.numAttributes()-1)
                    continue; //series not long enough to provide suffient gradient data
                
                Timer aTimer = new Timer("\ta="+a);
                
                BoTSW botsw = new BoTSW(n_b, a, kRanges[0]);  
                botsw.setSearchingForK(true);
                if (firstBuild) {
                    botsw.buildClassifier(data); //initial setup for these params 
                    fdData = botsw.fdData;
                    firstBuild = false; //save the guassian and keypoint data for all series,
                    //these are same regardless of (the searched) parameters for a given dataset, 
                    //so only compute once
                }
                else {
                    botsw.giveFeatureDiscoveryData(fdData);
                    botsw.buildClassifier(data);
                }
                
                //save the feature data (dependent on n_b and a) for reuse when searching for value of k
                Instances featureData = new Instances(botsw.clusterData); //constructor creates fresh copy
                
                boolean firstk = true;
                for (Integer k : kRanges) {       
                
                    Timer kTimer = new Timer("\t\tk="+k);
                    
                    if (firstk) //of this loop
                        firstk = false; //do nothing here, next loop go to the else
                    else {
                        botsw = new BoTSW(n_b, a, k); 
                        botsw.setSearchingForK(true);
                        botsw.giveFeatureData(featureData); 
                        botsw.buildClassifier(data);
                        //classifier now does not need to extract/describes features again
                        //simply clusters with new value of k
                    }
                        
                    int correct = 0; 
                    for (int i = 0; i < numSeries; ++i) {
                        double c = botsw.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                        if (c == data.get(i).classValue())
                            ++correct;
                    }
                    
                    double acc = (double)correct/(double)numSeries;     
                    //if not within correct threshold of the current max, dont bother storing at all
                    if (makesItIntoEnsemble(acc, maxAcc, minMaxAcc, classifiers.size())) {
                        BoTSWWindow bw = new BoTSWWindow(botsw, acc, data.relationName());
                        //bw.classifier.clean();

                        if (serOption == SerialiseOptions.STORE)
                            bw.store();
                        else if (serOption == SerialiseOptions.STORE_LOAD)
                            bw.storeAndClearClassifier();

                        classifiers.add(bw);

                        if (acc > maxAcc) {
                            maxAcc = acc;       
                            //get rid of any extras that dont fall within the new max threshold
                            Iterator<BoTSWWindow> it = classifiers.iterator();
                            while (it.hasNext()) {
                                BoTSWWindow b = it.next();
                                if (b.accuracy < maxAcc * correctThreshold) {
                                    if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD)
                                        b.deleteSerFile();
                                    it.remove();
                                }
                            }
                        }
                        
                        while (classifiers.size() > maxEnsembleSize) {
                            //cull the 'worst of the best' until back under the max size                            
                            int minAccInd = (int)findMinEnsembleAcc()[0];
                            
                            if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD)
                                classifiers.get(minAccInd).deleteSerFile();
                            classifiers.remove(minAccInd);
                        }
                        
                        minMaxAcc = findMinEnsembleAcc()[1]; //new 'worst of the best' acc
                    }
                    kTimer.printlnTimeSoFar();
                }
                aTimer.printlnTimeSoFar();
            }
            n_bTimer.printlnTimeSoFar();
        }
        
        if (trainCV) {
            int folds=setNumberOfFolds(data);
            OutFile of=new OutFile(trainCVPath);
            of.writeLine(data.relationName()+",BoTSWEnsemble,train");
           
            double[][] results = findEnsembleTrainAcc(data);
            of.writeLine(getParameters());
            of.writeLine(results[0][0]+"");
            ensembleCvAcc = results[0][0];
            for(int i=1;i<results[0].length;i++)
                of.writeLine(results[0][i]+","+results[1][i]);
            System.out.println("CV acc ="+results[0][0]);
        }
    }

    //[0] = index, [1] = acc
    private double[] findMinEnsembleAcc() {
        double minAcc = Double.MIN_VALUE;
        int minAccInd = 0;
        for (int i = 0; i < classifiers.size(); ++i) {
            double curacc = classifiers.get(i).accuracy;
            if (curacc < minAcc) {
                minAcc = curacc;
                minAccInd = i;
            }
        }
        
        return new double[] { minAccInd, minAcc };
    }
    
    private boolean makesItIntoEnsemble(double acc, double maxAcc, double minMaxAcc, int curEnsembleSize) {
        if (acc >= maxAcc * correctThreshold) {
            if (curEnsembleSize >= maxEnsembleSize)
                return acc > minMaxAcc;
            else 
                return true;
        }
        
        return false;
    }
    
    private double[][] findEnsembleTrainAcc(Instances data) throws Exception {
        
        double[][] results = new double[2][data.numInstances() + 1];
        
        double correct = 0; 
        for (int i = 0; i < data.numInstances(); ++i) {
            double c = classifyInstance(i, data.numClasses()); //classify series i, while ignoring its corresponding histogram i
            if (c == data.get(i).classValue())
                ++correct;
            
            results[0][i+1] = data.get(i).classValue();
            results[1][i+1] = c;
        }
        
        results[0][0] = correct / data.numInstances();
        //TODO fill results[1][0]
        
        return results;
    }
    
    public double getEnsembleCvAcc(){
        if(ensembleCvAcc>=0){
            return this.ensembleCvAcc;
        }
        
        try{
            return this.findEnsembleTrainAcc(train)[0][0];
        }catch(Exception e){
            e.printStackTrace();
        }
        return -1;
    }
    
    /**
     * Classify the train instance at index 'test', whilst ignoring the corresponding bags 
     * in each of the members of the ensemble, for use in CV of BoTSWEnsemble
     */
    public double classifyInstance(int test, int numclasses) throws Exception {
        double[] dist = distributionForInstance(test, numclasses);
        
        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) 
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
        
        return maxClass;
    }

    public double[] distributionForInstance(int test, int numclasses) throws Exception {
        double[] classHist = new double[numclasses];
        
        //get votes from all windows 
        double sum = 0;
        for (BoTSWWindow classifier : classifiers) {
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.load();
            double classification = classifier.classifyInstance(test);
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.clearClassifier();
            classHist[(int)classification]++;
            sum++;
        }
        
        if (sum != 0)
            for (int i = 0; i < classHist.length; ++i)
                classHist[i] /= sum;
        
        return classHist;
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
        for (BoTSWWindow classifier : classifiers) {
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.load();
            double classification = classifier.classifyInstance(instance);
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.clearClassifier();
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

    public static void main(String[] args) throws Exception{
        //Minimum working example
        String dataset = "BeetleFly";
        Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TEST.arff");
        
        Classifier c = new BoTSWEnsemble();
        c.buildClassifier(train);
        double accuracy = ClassifierTools.accuracy(test, c);
        
        System.out.println("BoTSWEnsemble accuracy on " + dataset + " fold 0 = " + accuracy);
        
        //Other examples/tests
//        detailedFold0Test(dataset);
//        resampleTest(dataset, 5);
    }
    
        public static void detailedFold0Test(String dset) {
        System.out.println("BoTSWEnsemble DetailedTest\n");
        try {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
            System.out.println(train.relationName());
            
            BoTSWEnsemble boss = new BoTSWEnsemble();
            
            //TRAINING
            System.out.println("Training starting");
            long start = System.nanoTime();
            boss.buildClassifier(train);
            double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Training done (" + trainTime + "s)");
            
            //RESULTS OF TRAINING
            System.out.println("Ensemble Size: " + boss.classifiers.size());
            System.out.println("Param sets: ");
            
            int count = 0;
            for (BoTSWWindow window : boss.classifiers)
                System.out.println(count++ + ": " + window.getNB() + " " + window.getA() + " " + window.getK() + " " + window.accuracy);

            //TESTING
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
        
    public static void resampleTest(String dset, int resamples) throws Exception {
        Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
         
        Classifier c = new BoTSWEnsemble();
         
        //c.setCVPath("C:\\tempproject\\BOSSEnsembleCVtest.csv");
         
        double [] accs = new double[resamples];
         
        for(int i=0;i<resamples;i++){
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
            
            c.buildClassifier(data[0]);
            accs[i]= ClassifierTools.accuracy(data[1], c);
            
            if (i==0)
                System.out.print(accs[i]);
            else 
                System.out.print("," + accs[i]);
        }
         
        double mean = 0;
        for(int i=0;i<resamples;i++)
            mean += accs[i];
        mean/=resamples;
         
        System.out.println("\n\nBOSSEnsemble mean acc over " + resamples + " resamples: " + mean);
    }
}
