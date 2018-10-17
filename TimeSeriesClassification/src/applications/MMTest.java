/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package applications;

import development.ExperimentsKaggle;
import development.MultipleClassifierEvaluation;

/**
 *
 * @author pfm15hbu
 */
public class MMTest {
    
    static int maxFolds = 20;
    
    public static void main(String[] args) throws Exception {
        classifierEvaluation(args);
        //experiment(args);
        //kaggleExperiment(args);
    }
    
    static void classifierEvaluation(String[] args) throws Exception{
        MultipleClassifierEvaluation x = new MultipleClassifierEvaluation("C:/UEAMachineLearning/Tests/TSCMultivariate/", "MultivariateTests", 20);
        x.setDatasets(new String[] {"LSST1"});
        x.readInClassifiers(new String[] {"DTW_A", "DTW_D", "DTW_I", "ED_D", "ED_I", "RandF", "RotF", "SVML", "XGBoost"}, "C:/UEAMachineLearning/Tests/TSCMultivariate/");
        x.runComparison(); 
    }
    
    static void experiment(String[] args){
        
    }
    
    static void kaggleExperiment(String[] args) throws Exception{
        for(int fold = 1; fold < maxFolds+1; fold++){
            String[] newArgs = new String[9];
            newArgs[0] = "C:/UEAMachineLearning/Datasets/Kaggle/PLAsTiCCAstronomicalClassification/";
            newArgs[1] = "C:/UEAMachineLearning/Datasets/Kaggle/PLAsTiCCAstronomicalClassification/Results/";
            newArgs[2] = "false";
            newArgs[3] = "TunedXGBoost";
            newArgs[4] = "LSST1";
            newArgs[5] = Integer.toString(fold);
            newArgs[6] = "false";
            newArgs[7] = "0";
            newArgs[8] = "false";
            
            ExperimentsKaggle.main(newArgs);
        }
    }
}
