/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Rise2;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.filters.ACF;
import timeseriesweka.filters.FFT;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author cjr13geu
 */
public class RiseV2 implements Classifier{
    
    private Random random = null;
    private Classifier baseClassifier = null;
    private Classifier[] baseClassifiers = null;
    private int[][] startEndPoints = null;
    private int numClassifiers = 0;
    private int interval = 0;
    private String relationName = null;
    private Filter filter;
    private enum Filter{PS,ACF,FFT,PS_ACF};
    private FFT fft;
    private long seed = 0;
    private Boolean buildFromSavedData;
    private Instances testInstances = null;
    private int testClassificationIndex = 0;
    private int minimumIntervalLength = 2;
    private int maximumIntervalLength = 50;
    
    
    public RiseV2(Long seed){
        this.seed = seed;
        random = new Random(seed);
        initialise();
    }
    
    public RiseV2(){
        random = new Random();
        initialise();
    }
    
    private void initialise(){
        numClassifiers = 50;
        setBaseClassifier();
        filter = Filter.PS;
        fft = new FFT();
        startEndPoints = new int[numClassifiers][2];
        buildFromSavedData = false;
    }
    
    private void setBaseClassifier(){
        baseClassifier = new RandomTree();
    }
    
    public void setBaseClassifier(Classifier classifier){
        baseClassifier = classifier;
    }
    
    public void setMinimumIntervalLength(int length){
        minimumIntervalLength = length;
    }
    
    public void setMaximumIntervalLength(int length){
        maximumIntervalLength = length;
    }
    
    public void setNumClassifiers(int numClassifiers){
        this.numClassifiers = numClassifiers;
        startEndPoints = new int[numClassifiers][2];
    }
    
    public void buildFromSavedData(Boolean buildFromSavedData){
        this.buildFromSavedData = buildFromSavedData;
    }
    
    public void setTransformType(String s){
        
        String str=s.toUpperCase();
        switch(str){
            case "ACF": case "AFC": case "AUTOCORRELATION":
                filter = Filter.ACF;                
                break;
            case "PS": case "POWERSPECTRUM":
                filter = Filter.PS;
                break;
            case "PS_ACF": case "ACF_PS": case "BOTH":
                filter = Filter.PS_ACF;
                break;       
        }
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(maximumIntervalLength > data.get(0).numAttributes()-1 || maximumIntervalLength <= 0){
            maximumIntervalLength = data.get(0).numAttributes()-1;
        }
        if(minimumIntervalLength >= data.get(0).numAttributes()-1 || minimumIntervalLength <= 0){
            minimumIntervalLength = 2;
        }
        
        testClassificationIndex = 0;
        baseClassifiers = new Classifier[numClassifiers];
        //startEndPoints[x][0] = start points.
        //startEndPoints[x][1] = end points.

        if (!buildFromSavedData)
            initialiseStartEndArray(data);
        
        for (int i = 0; i < numClassifiers; i++) {
            
            Instances intervalInstances = null;
            
            if(!buildFromSavedData)
                intervalInstances = produceIntervalInstances(data, i);
            else
                intervalInstances = ClassifierTools.loadData("RISE/Training Data/Fold " + (int)seed + "/Classifier " + i);
            //TRAIN CLASSIFIERS.
            baseClassifiers[i] = AbstractClassifier.makeCopy(baseClassifier);
            baseClassifiers[i].buildClassifier(intervalInstances);
        }
    }
    
    private void initialiseStartEndArray(Instances instances){
        for (int i = 0; i < numClassifiers; i++) {
            //Produce start and end values for interval, can include whole series.
            do{
                startEndPoints[i][0] = random.nextInt(instances.numAttributes()-1); 
                startEndPoints[i][1] = random.nextInt((instances.numAttributes()) - startEndPoints[i][0]) + startEndPoints[i][0];
                interval = startEndPoints[i][1] - startEndPoints[i][0];     
            }while(interval < minimumIntervalLength || interval > maximumIntervalLength);
            //System.out.println(interval);
        }
    }
    
    private Instances produceTransform(Instances instances){
        
        Instances temp = null;
        
        switch(filter){
            case ACF:
                temp = ACF.formChangeCombo(instances);
                break;
            case PS: 
                try {
                    fft.useFFT();
                    
                    temp = fft.process(instances);
                } catch (Exception ex) {
                    Logger.getLogger(RiseV2.class.getName()).log(Level.SEVERE, null, ex);
                }
                break;
            case PS_ACF: 
                temp = combinedPSACF(instances);
                break;
        }
        return temp;
    }
    
    private Instances combinedPSACF(Instances instances){
        
        Instances combo=ACF.formChangeCombo(instances);
        Instances temp = null;
        try {
            temp = fft.process(instances);
        } catch (Exception ex) {
            Logger.getLogger(RiseV2.class.getName()).log(Level.SEVERE, null, ex);
        }
        combo.setClassIndex(-1);
        combo.deleteAttributeAt(combo.numAttributes()-1); 
        combo = Instances.mergeInstances(combo, temp);
        combo.setClassIndex(combo.numAttributes()-1);
        
        return combo;        
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {

        double[]distribution = distributionForInstance(instance);

        int maxVote=0;
        for(int i = 1; i < distribution.length; i++)
            if(distribution[i] > distribution[maxVote])
                maxVote = i;
        return maxVote;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        
        double[]distribution = new double[instance.numClasses()];
        
        for (int i = 0; i < numClassifiers; i++) {
            int classVal = 0;
            
            if (!buildFromSavedData) {
                classVal = (int) baseClassifiers[i].classifyInstance(produceIntervalInstance(instance, i));
            }else{
                testInstances = ClassifierTools.loadData("RISE/Test Data/Fold " + (int)seed + "/Classifier " + i);
                classVal = (int) baseClassifiers[i].classifyInstance(testInstances.get(testClassificationIndex));
            }        
            distribution[classVal]++;    
        }
        
        for(int i = 0 ; i < distribution.length; i++)
            distribution[i] /= baseClassifiers.length;
        
        if (buildFromSavedData)
            testClassificationIndex++;
        
        return distribution;
    }
    
    private Instance produceIntervalInstance(Instance instance, int classifierNum){
        
        ArrayList<Attribute>attributes = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attributes.add(instance.attribute(i));
        }
        Instances intervalInstances = new Instances(relationName, attributes, 1);
        intervalInstances.add(instance);
        intervalInstances.setClassIndex(instance.numAttributes()-1);
        intervalInstances = produceIntervalInstances(intervalInstances, classifierNum);
        
        return intervalInstances.firstInstance();
    }
    
    private Instances produceIntervalInstances(Instances instances, int classifierNum){

        //POPULATE INTERVAL INSTANCES. 
        //Create and populate attribute information based on interval, class attribute is an addition.
        ArrayList<Attribute>attributes = new ArrayList<>();
        for (int i = startEndPoints[classifierNum][0]; i < startEndPoints[classifierNum][1]; i++) {
            attributes.add(instances.attribute(i));
        }
        attributes.add(instances.attribute(instances.numAttributes()-1));

        //Create new Instances to hold intervals.
        relationName = instances.relationName();
        Instances intervalInstances = new Instances(relationName, attributes, instances.size());

        for (int i = 0; i < instances.size(); i++) {
            //Produce intervals from input instances, additional attribute needed to accomidate class value.
            double[] temp = Arrays.copyOfRange(instances.get(i).toDoubleArray(), startEndPoints[classifierNum][0], startEndPoints[classifierNum][1] + 1);
            DenseInstance instance = new DenseInstance(temp.length);
            instance.replaceMissingValues(temp);
            instance.setValue(temp.length-1, instances.get(i).classValue());
            intervalInstances.add(instance);     
        }
        intervalInstances.setClassIndex(intervalInstances.numAttributes()-1);
        
        intervalInstances = produceTransform(intervalInstances);
        
        return intervalInstances;
    }

    public void createIntervalInstancesARFF(Instances training, Instances test){
        initialiseStartEndArray(training);
        
        for (int i = 0; i < numClassifiers; i++) {
            if (!(new File("RISE/Training Data/Fold " + (int)seed + "/Classifier " + i + ".arff").isFile())) {
                ClassifierTools.saveDataset(produceIntervalInstances(training, i), "RISE/Training Data/Fold " + (int)seed + "/Classifier " + i);
            }
            if (!(new File("RISE/Test Data/Fold " + (int)seed + "/Classifier " + i + ".arff").isFile())) {
                ClassifierTools.saveDataset(produceIntervalInstances(test, i), "RISE/Test Data/Fold " + (int)seed + "/Classifier " + i);
            }  
        }   
    }
    
    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args) throws Exception {
        
        Instances all = ClassifierTools.loadData("/gpfs/home/cjr13geu/Datasets/MosquitoDatasets/Unaltered/Time/Time.arff");
        
        //LOCAL TESTING PATH.
        //Instances all = ClassifierTools.loadData("U:\\Documents\\NetBeansProjects\\Testing PhD\\RISE Testing\\Data\\Cancer.arff");
        
        Instances[] instances;
        
        RiseV2 rise = new RiseV2((long) Integer.parseInt(args[0])-1);
        instances = InstanceTools.resampleInstances(all, Integer.parseInt(args[0])-1, .5);
        rise.createIntervalInstancesARFF(instances[0], instances[1]);
        
        
        //rise.setNumClassifiers(50);
        
        
        
        
        
        //rise.buildFromSavedData(false);
        
        //rise.buildClassifier(train);
        
        //ArrayList<double[]> testDistributions = new ArrayList<>();
        //ArrayList<Double> testClassifications = new ArrayList<>();
        
        //for (int i = 0; i < test.size(); i++) {
        //    testDistributions.add(rise.distributionForInstance(test.get(i)));
            //testClassifications.add(rise.classifyInstance(test.get(i)));
        //}
        
        //double accuracy = 0;
        //for (int i = 0; i < test.size(); i++) {
            
            //if(testClassifications.get(i) == test.get(i).classValue())
            //    accuracy++;
            
        //    System.out.println(testDistributions.get(i)[0] + " - " + testDistributions.get(i)[1]);
        //}
       
        //System.out.println("Accuracy: " + accuracy/test.size());
    }
    
}