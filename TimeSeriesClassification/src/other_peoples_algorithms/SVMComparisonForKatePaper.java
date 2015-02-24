/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package other_peoples_algorithms;

import development.DataSets;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SVMComparisonForKatePaper {
     
    
    public static void main(String[] args){
        SMO smo=new SMO();
        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(1);
        smo.setKernel(kernel);
        String path ="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
            OutFile of = new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Other Peoples Published Results\\Kate DTW\\SVM_Results.csv");
        for(String s:DataSets.ucrNames){
            Instances train = ClassifierTools.loadData(path+s+"\\"+s+"_TRAIN");
            Instances test = ClassifierTools.loadData(path+s+"\\"+s+"_TEST");
            
            Evaluation eval;
            try {
                eval = new Evaluation(train);
                eval.crossValidateModel(smo, train, 10, new Random(1));
                double a1=eval.correct()/(double)train.numInstances();
                eval = new Evaluation(train);
                smo=new SMO();
                kernel.setExponent(2);
                smo.setKernel(kernel);
                eval.crossValidateModel(smo, train, 10, new Random(1));
                double a2=eval.correct()/(double)train.numInstances();
                kernel.setExponent(3);
                smo=new SMO();
                smo.setKernel(kernel);
                eval = new Evaluation(train);
                eval.crossValidateModel(smo, train, 10, new Random(1));
                double a3=eval.correct()/(double)train.numInstances();
                int exponent=1;
                double best=a1;
                if(a2>best){
                    best=a2;
                    exponent=2;
                }
                if(a3>best){
                    best=a3;
                    exponent=3;
                }
                DecimalFormat df = new DecimalFormat("###.###");
                kernel.setExponent(exponent);
                smo=new SMO();
                smo.setKernel(kernel);
                double a=ClassifierTools.singleTrainTestSplitAccuracy(smo, train, test);
                System.out.println(s+","+exponent+","+df.format(a)+" linear ="+df.format(a1)+" quad ="+df.format(a2)+" tri ="+df.format(a3));
                of.writeLine(s+","+exponent+","+a);
                
            } catch (Exception ex) {
                Logger.getLogger(SVMComparisonForKatePaper.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
}
    
    
}
