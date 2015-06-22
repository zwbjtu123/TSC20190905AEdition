package grabocka_reproduction;

import static utilities.InstanceTools.create2DMatrixFromInstances;
import static utilities.StatisticalUtilities.Normalize2D;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;

public class GrabockaReproduction {

    @SuppressWarnings("empty-statement")
    public static void main(String[] args) {
        experiment1(args);
    }

    public static void experiment1(String[] args) {
        String dir = "75 Data sets for Elastic Ensemble DAMI Paper";

        File fDir = new File(dir);
        File[] ds = fDir.listFiles();

        if (args.length < 1 || args[0] == null) return;

        int index = Integer.parseInt(args[0]) - 1;
        System.out.println(ds[index]);
        
        try (PrintWriter outFile = new PrintWriter(new FileWriter("grabocka" + index + ".txt"))) {
            runAlgorithm(ds[index], outFile);
        } catch (IOException ex) {
            System.out.println("IOException " + ex);
        }
    }

    public static void runAlgorithm(File f, PrintWriter outFile) {
        String temp = f.toString() + File.separator + f.getName();
        String trainSetPath = temp + "_TRAIN";
        String testSetPath = temp + "_TEST";

        double errorRate = 1.0;
        int pos = -1;
        int noFolds = 3;
        double sumError = 0;

        // load the train and test sets
        Instances train = loadData(trainSetPath);
        train.randomize(new Random());
        
        //should we stratify?
        double[] lambdaW = {0.01, 0.1};

        double[] L = {0.1, 0.2};

        double[] R = {2, 3};

        ArrayList<double[]> combinations = new ArrayList<>();

        double[][] data= {L, R, lambdaW};

        combinationBuilder(data, 0, null, combinations);

        String dsName = f.getName();

        String output = "dataSet,L,R,lamdaW,error";
        System.out.println(output);
        outFile.println(output);
        
        long startTime, endTime;
        double sumTime = 0;
        //do each combiantion of params.
        for (int i = 0; i < combinations.size(); i++) {
            double[] params = combinations.get(i);

            //build our test and train sets. for cross-validation.
            for (int j = 0; j < noFolds; j++) {
                Instances trainCV = train.trainCV(noFolds, j);
                Instances testCV = train.testCV(noFolds, j);

                //time our experiment
                startTime = System.currentTimeMillis();

                sumError += createLearnShapeletsGeneralized(trainCV, testCV, dsName, params, outFile);

                endTime = System.currentTimeMillis();

                //add our sum time up.
                sumTime += (endTime - startTime);
            }

            //find mean Error
            double avgCVError = sumError / noFolds;
            double avgCVtime = sumTime / noFolds;

            String p = Arrays.toString(params);
            p = p.replace("[", "");
            p = p.replace("]", "");
            output = dsName + ","
                    + p + ","
                    + String.valueOf(avgCVError)
                    + ","
                    + avgCVtime; //avg ourTime

            System.out.println(output);
            outFile.println(output);

            //store best errorRate and pos.
            if (avgCVError < errorRate) {
                errorRate = avgCVError;
                pos = i;
            }

            sumError = 0;
            sumTime = 0;
        }

        Instances test = loadData(testSetPath);

        //time our experiment
        startTime = System.currentTimeMillis();
        errorRate = createLearnShapeletsGeneralized(train, test, dsName, combinations.get(pos), outFile);
        endTime = System.currentTimeMillis();

        String p = Arrays.toString(combinations.get(pos));
        p = p.replace("[", "");
        p = p.replace("]", "");
        output = dsName + ","
                + p + ","
                + String.valueOf(errorRate)
                + ","
                + (endTime - startTime); //avg ourTime

        System.out.println(output);
        outFile.println(output);
    }

    public static void combinationBuilder(double[][] data, int pointer, double[] currentArgs, ArrayList<double[]> combinations) {
        //base case
        if (pointer >= data.length) {
            //we've traversed through our Array of data.
            //and we need to do something.
            combinations.add(currentArgs);
        } else {
            int currentArgsSize = (currentArgs != null) ? currentArgs.length : 0;

            //fill up our Array with values from our data points. 
            for (int i = 0; i < data[pointer].length; i++) {
                double[] temp = new double[currentArgsSize + 1];
                if (currentArgs != null) {
                    System.arraycopy(currentArgs, 0, temp, 0, currentArgsSize);
                }

                temp[temp.length - 1] = data[pointer][i];

                combinationBuilder(data, pointer + 1, temp, combinations);
            }
        }
    }

    //params come in the order = {K,lambdaW,maxEpochs,alpha,eta,L,R};
    public static double createLearnShapeletsGeneralized(Instances train, Instances test, String dsName, double[] params, PrintWriter outFile) {

        double errorRate = 0;
        
        //predictor variables T
        double[][] T = create2DMatrixFromInstances(train,test);
        //normalise all the series in the matrix.
        //class value at the end of the series.
        Normalize2D(T, true);
        
        //outcome variable O
        double [][] O = create2DMatrixFromInstances(train,test);
        //normalise all the series in the matrix.
        //class value at the end of the series.
        Normalize2D(O, true);
        
        LearnShapelets lsg = new LearnShapelets();
        //{PercentageOfSeriesLength,shapeletLengthScale, weights}; 
        lsg.percentageOfSeriesLength = (int) params[0];
        lsg.shapeletLengthScale = (int) params[1];
        lsg.lambdaW = params[2];
        double accuracy = 0;
        try {
            lsg.buildClassifier(train);
            accuracy = ClassifierTools.accuracy(test, lsg);
        
        } catch (Exception ex) {
            System.out.println("Error");
        }

        //return error rate
        return 1-accuracy;
    }
    

    public static Instances loadData(String fullPath) {
        Instances d = null;
        FileReader r;
        try {
            r = new FileReader(fullPath + ".arff");
            d = new Instances(r);
            d.setClassIndex(d.numAttributes() - 1);
        } catch (IOException e) {
            System.out.println("Unable to load data on path " + fullPath + " Exception thrown =" + e);
        }
        return d;
    }

}
