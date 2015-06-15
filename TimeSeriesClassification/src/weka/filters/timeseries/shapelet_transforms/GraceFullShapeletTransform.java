/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms;

import AaronTest.LocalInfo;
import development.DataSets;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.shapelet.Shapelet;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.removeSelfSimilar;

/**
 *
 * @author Aaron
 */
public class GraceFullShapeletTransform extends FullShapeletTransform {

    int currentSeries = 0;

    String seriesShapeletsFilePath;

    public void setSeries(int i) {
        currentSeries = i;
    }

    /**
     * The main logic of the filter; when called for the first time, k shapelets
     * are extracted from the input Instances 'data'. The input 'data' is
     * transformed by the k shapelets, and the filtered data is returned as an
     * output.
     * <p>
     * If called multiple times, shapelet extraction DOES NOT take place again;
     * once k shapelets are established from the initial call to process(), the
     * k shapelets are used to transform subsequent Instances.
     * <p>
     * Intended use:
     * <p>
     * 1. Extract k shapelets from raw training data to build filter;
     * <p>
     * 2. Use the filter to transform the raw training data into transformed
     * training data;
     * <p>
     * 3. Use the filter to transform the raw testing data into transformed
     * testing data (e.g. filter never extracts shapelets from training data,
     * therefore avoiding bias);
     * <p>
     * 4. Build a classifier using transformed training data, perform
     * classification on transformed test data.
     *
     * @param data the input data to be transformed (and to find the shapelets
     * if this is the first run)
     * @return the transformed representation of data, according to the
     * distances from each instance to each of the k shapelets
     */
    @Override
    public Instances process(Instances data) throws IllegalArgumentException {
        //check the input data is correct and assess whether the filter has been setup correctly.
        inputCheck(data);

        //setup classsValue
        classValue.init(data);
        //setup subseqDistance
        subseqDistance.init(data);

        //checks if the shapelets haven't been found yet, finds them if it needs too.
        if (!shapeletsTrained) {
            trainShapelets(data);
            shapeletsTrained = false; //set the shapelets Trained to false, because we'll set it to true once all the sub code has been finished.
            outputPrint("Partially Built the shapelet Set");
            return null;
        }

        //build the transformed dataset with the shapelets we've found either on this data, or the previous training data
        return buildTansformedDataset(data, shapelets);
    }

    /**
     * protected method for extracting k shapelets.
     *
     * @param data the data that the shapelets will be taken from
     * @return an ArrayList of FullShapeletTransform objects in order of their
     * fitness (by infoGain, seperationGap then shortest length)
     */
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data) {
        ArrayList<Shapelet> kShapelets = new ArrayList<>();
        ArrayList<Shapelet> seriesShapelets;                                    // temp store of all shapelets for each time series

        //for all time series
        outputPrint("Processing data: ");

        outputPrint("data : " + currentSeries);

        double[] wholeCandidate = data.get(currentSeries).toDoubleArray();

        //changed to pass in the worst of the K-Shapelets.
        Shapelet worstKShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;

        //set the series we're working with.
        subseqDistance.setSeries(currentSeries);
        //set the clas value of the series we're working with.
        classValue.setShapeletValue(data.get(currentSeries));

        seriesShapelets = findShapeletCandidates(data, currentSeries, wholeCandidate, worstKShapelet);

        Comparator comp = useSeparationGap ? new Shapelet.ReverseSeparationGap() : new Shapelet.ReverseOrder();
        Collections.sort(seriesShapelets, comp);

        seriesShapelets = removeSelfSimilar(seriesShapelets);

        kShapelets = combine(numShapelets, kShapelets, seriesShapelets);

        this.numShapelets = kShapelets.size();

        recordShapelets(kShapelets, getSubShapeletFileName(currentSeries));
        printShapelets(kShapelets);

        return kShapelets;
    }
    
    private String getSubShapeletFileName(int i)
    {
        File f = new File(this.ouputFileLocation);
        String str = f.getName();
        str = str.substring(0, str.lastIndexOf('.'));
        return str + i + ".csv";
    }

    public Instances processFromSubFile(Instances train) {
        File f = new File(this.ouputFileLocation);

        ArrayList<Shapelet> kShapelets = new ArrayList<>();
        ArrayList<Shapelet> seriesShapelets;
        int i=0;
        while(true) {
            seriesShapelets = readShapeletsFromFile(getSubShapeletFileName(i++));
            if(seriesShapelets == null) break; //we've reached the end of the files.
            kShapelets = combine(numShapelets, kShapelets, seriesShapelets);
        }
        
        numShapelets = kShapelets.size();
        shapelets = kShapelets;
        shapeletsTrained = true;

        return buildTansformedDataset(train, shapelets);
    }

    public static ArrayList<Shapelet> readShapeletsFromFile(String shapeletLocation){
        try {
            ArrayList<Shapelet> shapelets = new ArrayList<>();
            File f = new File(shapeletLocation);
            
            Scanner sc = new Scanner(f);
            String[] lineData;
            double[] timeSeries;
            double qualityValue;
            int seriesId,startPos, i;
            Shapelet s;
            
            while (sc.hasNextLine()) {
                lineData = sc.nextLine().split(",");
                
                qualityValue = lineData.length >= 1 ? Double.parseDouble(lineData[0]) : 0.0;
                seriesId = lineData.length >= 2 ? Integer.parseInt(lineData[1]) : 0;
                startPos = lineData.length >= 3 ? Integer.parseInt(lineData[2]) : 0;
                
                //first three elements of the data have been accounted, next line is the contents of the shapelet.
                lineData = sc.nextLine().split(",");
                
                timeSeries = new double[lineData.length];
                for (i = 0; i < timeSeries.length; ++i) {
                    timeSeries[i] = Double.parseDouble(lineData[i]);
                }
                
                shapelets.add(new Shapelet(timeSeries, qualityValue, seriesId, startPos));
            }
            
            return shapelets;
            
        } catch (FileNotFoundException ex) {
            System.out.println("File Not Found");
            return null;
        }
    }

    //memUsage is in MB.
    public static void buildGraceBSUB(String filePath, String savePath, int arraySize, String userName, String jarPath, String jobName, String queue, int memUsage) {
        try {
            //create the directory and the files.
            File f = new File(savePath + ".bsub");
            f.getParentFile().mkdirs();
            f.createNewFile();

            try (PrintWriter pw = new PrintWriter(f)) {
                pw.println("#!/bin/csh");
                pw.println("#BSUB -q " + queue);
                pw.println("#BSUB -J " + jobName + "[1-" + arraySize + "]"); //+1 because we have to start at 1.
                pw.println("#BSUB -cwd /gpfs/sys/" + userName + "/" + jarPath);
                pw.println("#BSUB -oo " + jobName + "_%I.out");
                pw.println("#BSUB -R \"rusage[mem=" + memUsage + "]\"");
                pw.println("#BSUB -M " + (memUsage * 1.2)); //give ourselves a 20% wiggle room.
                pw.println(". /etc/profile");
                pw.println("module add java/jdk/1.7.0_13");
                pw.println("java -jar -Xmx" + memUsage + "m TimeSeriesClassification.jar search $LSB_JOBINDEX " + filePath + " " + savePath);

                pw.println();
                pw.println();
                pw.println("#BSUB -q " + queue);
                pw.println("#BSUB -J " + jobName); //+1 because we have to start at 1.
                pw.println("#BSUB -cwd /gpfs/sys/" + userName + "/" + jarPath);
                pw.println("#BSUB -oo " + jobName + "_%I.out");
                pw.println("#BSUB -R \"rusage[mem=" + memUsage + "]\"");
                pw.println("#BSUB -M " + (memUsage * 1.2)); //give ourselves a 20% wiggle room.
                pw.println(". /etc/profile");
                pw.println("module add java/jdk/1.7.0_13");
                pw.println("java -jar -Xmx" + memUsage + "m TimeSeriesClassification.jar combine " + filePath + " " + savePath);
            }
        } catch (IOException ex) {
            System.out.println("Failed to create file " + ex);
        }
    }

    public static void main(String[] args) {
        
        //we assume at the file path you have two files which have _TRAIN and _TEST attached to them.
        // .jar search 1 ../../Time-Series-Datasets/Adiac/Adiac

        //.jar combine  ../../Time-Series-Datasets/Adiac/Adiac ../../Time-Series-transforms/Adiac/Adiac
        GraceFullShapeletTransform st = new GraceFullShapeletTransform();

        if(args[0].equalsIgnoreCase("search"))
        {
            int number = Integer.parseInt(args[1]);
            Instances train = utilities.ClassifierTools.loadData(args[2]+"_TRAIN");
        
            //set the params for your transform. length, shapelets etc.
            st.setLogOutputFile(args[2] + ".csv");
            st.setNumberOfShapelets(train.numInstances()*10);

            //partial training.
            st.setSeries(number-1);
            st.process(train);
        }
        
        else if(args[0].equalsIgnoreCase("combine"))
        {
            st.setLogOutputFile(args[1] + ".csv");
     
            Instances train = utilities.ClassifierTools.loadData(args[1]+"_TRAIN");
            Instances test = utilities.ClassifierTools.loadData(args[1]+"_TEST");
            
            LocalInfo.saveDataset(st.processFromSubFile(train), args[2] + "_TRAIN");
            LocalInfo.saveDataset(st.process(test), args[2] + "_TEST");
        
        }
        
    }
    
    public static void test()
    {
        final String ucrLocation = "../../time-series-datasets/TSC Problems";
        final String transformLocation = "../../";

        String fileExtension = File.separator + DataSets.ucrSmall[0] + File.separator + DataSets.ucrSmall[0];

        Instances train = utilities.ClassifierTools.loadData(ucrLocation + fileExtension + "_TRAIN");
        Instances test = utilities.ClassifierTools.loadData(ucrLocation + fileExtension + "_TEST");

        //first run: build the BSUB.
        //GraceFullShapeletTransform.buildGraceBSUB("../../"+DataSets.ucrSmall[0], train.numInstances(), "raj09hxu", "SamplingExperiments/dist", "samplingExperiments", "long", 1000);
        

        GraceFullShapeletTransform st = new GraceFullShapeletTransform();
        st.setNumberOfShapelets(train.numInstances()*10);
        st.setLogOutputFile(DataSets.ucrSmall[0] + ".csv");

        //set the params for your transform. length, shapelets etc.
        
        //second run: using the BSUB. for the cluster
        //st.setSeries(Integer.parseInt(args[0])-1);
        //st.process(train);
        

        //third run: for your own machine. this will build the datasets.
        String classifierDir = File.separator + st.getClass().getSimpleName() + fileExtension;
        String savePath = transformLocation + classifierDir;

        LocalInfo.saveDataset(st.processFromSubFile(train), savePath + "_TRAIN");
        LocalInfo.saveDataset(st.process(test), savePath + "_TEST");
        /**/
    }

}
