/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package other_peoples_algorithms;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.SAX;

/**
 *
 * @author raj09hxu
 */
public class FastShapelets implements Classifier {

    SAX sax;
    int saxCardinality = 4;
    int saxWordLength = 16;

    int numberRandomIterations;

    Instances saxData;
    Random rand;

    Map<String,int[]> collisionTable;
    Map<String, int[]> closeToRef;

    public FastShapelets() {
        sax = new SAX();
        sax.setAlphabetSize(saxCardinality);
        sax.setNumIntervals(saxWordLength);
        rand = new Random();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        //sax our data.
        saxData = sax.process(data);

        //set our random iterations
        numberRandomIterations = 5;

        String[] saxList;

        //I've changed starting length to 5, that's the one they use as minimum.
        for (int len = 5; len < saxData.numAttributes(); len++) {
            saxList = createSAXList(saxData, len);

            //create collision table for this word length.
            collisionTable = new HashMap<>();
            initCollisionTable(saxList);
            
            closeToRef = new TreeMap();
            
            for (int i = 0; i < numberRandomIterations; i++) {
                randProjection(saxList, len);
            }
            
            constructCloseToRef();
            
            for(String word : collisionTable.keySet()){
                System.out.println(word + " : "+Arrays.toString(collisionTable.get(word)));
            }
            
            for(String word : closeToRef.keySet()){
                System.out.println(word + " : "+Arrays.toString(closeToRef.get(word)));
            }

        }

        //transform data into SAX.
        //saxData = sax.process(data);
    }
    
    private void constructCloseToRef()
    {
        for (Map.Entry<String, int[]> entry : collisionTable.entrySet()) {
            int[] distribution = entry.getValue();
            int[] classes = closeToRef.get(entry.getKey());
            //if it doesn't exist. init it.
            if(classes == null){
                classes = new int[saxData.numClasses()];
            }

            for(int i=0; i<distribution.length; i++){
                //find the class index for our value in the distibution.
                //update the classes array with the distribution value.
                int classIndex = (int)saxData.get(i).classValue();
                classes[classIndex]+= distribution[i];
            }

            closeToRef.put(entry.getKey(), classes);
    }
    }

    //generate a set of SAX shapelets of a set length.
    private static String[] createSAXList(Instances data, int len) {
        String[] saxList = new String[data.size() * (data.numAttributes() - len)];

        int k = 0;
        for (int i = 0; i < data.size(); i++) {
            double[] wholeCandidate = data.instance(i).toDoubleArray();
            for (int start = 0; start < wholeCandidate.length - len; start++) {
                String candidate = "";
                for (int j = 0; j < len; j++) {
                    candidate += data.instance(i).stringValue(start + j);
                }
                saxList[k++] = candidate;
            }
        }
        
        return saxList;
    }
    
    //we need to initalise our keys before we do random projection.
    private void initCollisionTable(String[] saxList)
    {
        for (String saxWord : saxList) {
            int[] distribution = collisionTable.get(saxWord);
            
            //if it doesn't exist, init the memory and put it in.
            if(distribution == null){
                distribution = new int[saxData.numInstances()];
                collisionTable.put(saxWord, distribution);
            }
        }
    }

    private void randProjection(String[] saxList, int length) {
        //number of rows is equal to the number of subsequence sax words we've created
        
        //we minus 2 off length so we can have a mask size of 2 and not overflow
        int maskSize = 1;
        int maskPos = (int) (rand.nextDouble() * (length-maskSize));
        
        
        /*int sax_len = 16;
        /// Make w and sax_len both integer
        int w = (int)Math.ceil(1.0*length/sax_len);
        sax_len = (int)Math.ceil(1.0*length/w);
        
        //percent mask is set to 25% of the words legnth.
        int num_mask = (int)Math.ceil(0.25 * sax_len);*/
        
        for (int pos = 0; pos < saxList.length; pos++) {

            String saxWord = saxList[pos];
            
            //calculate our class based on our ordering
            int series = pos / (saxData.numAttributes() - saxWord.length());
            
            //random masking on the saxWord. we replace the character and character + maskSize from our maskPos with [abcd]{maskSize} for regex.
            String maskedWord = saxWord.substring(0, maskPos) + "[abcd]{"+maskSize+"}" + saxWord.substring(maskPos+maskSize); 
            
            //map our masked words to the set of our real words.
            for (Map.Entry<String, int[]> entry : collisionTable.entrySet()) {
                if (entry.getKey().matches(maskedWord)) {
                    entry.getValue()[series]++;
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void main(String[] args) throws Exception {
        final String dotdotSlash = ".." + File.separator;
        String datasetName = "ItalyPowerDemand";
        String datasetLocation = dotdotSlash + dotdotSlash + "75 Data sets for Elastic Ensemble DAMI Paper" + File.separator + datasetName + File.separator + datasetName;

        Instances train = utilities.ClassifierTools.loadData(datasetLocation + "_TRAIN");

        FastShapelets fs = new FastShapelets();

        //try {
            fs.buildClassifier(train);

        //} catch (Exception ex) {
        //    System.out.println("Exception " + ex);
        //}
    }
    


}
