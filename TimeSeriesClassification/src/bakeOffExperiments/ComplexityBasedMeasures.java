/*
CID and RCPD 


 */
package bakeOffExperiments;

import static bakeOffExperiments.Experiments.threadedSingleClassifier;
import fileIO.OutFile;
import tsc_algorithms.NN_CID;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

/**
 *
 * @author ajb
 */
public class ComplexityBasedMeasures {
    
    public static void main(String[] args) throws Exception{
        Classifier c = new NN_CID();
        String s="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\complexity\\";
        OutFile res;

        System.out.println("******** CID Euclidean *******************");
        res = new OutFile(s+"CID_ED.csv");
        threadedSingleClassifier(c,res);

        System.out.println("******** CID DTW *******************");
        c= new NN_CID();
        ((NN_CID)c).useDTW();
        res = new OutFile(s+"CID_DTW.csv");
        threadedSingleClassifier(c,res);
        
    }
    
}
