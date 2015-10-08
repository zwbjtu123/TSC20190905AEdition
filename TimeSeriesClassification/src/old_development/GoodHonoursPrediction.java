/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package old_development;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class GoodHonoursPrediction {
    
    public static void main(String[] args){
        Instances data = ClassifierTools.loadData("C:\\Admin\\Perfomance Analysis\\GoodHonsClassification");
        RandomForest rf= new RandomForest();
        double[][] a=ClassifierTools.crossValidationWithStats(rf, data, data.numInstances());
        System.out.println(" Random forest LOOCV accuracy ="+a[0][0]);
        J48 tree = new J48();
        a=ClassifierTools.crossValidationWithStats(tree, data, data.numInstances());
        System.out.println(" C4.5 LOOCV accuracy ="+a[0][0]);
        IBk knn= new IBk(11);
        knn.setCrossValidate(true);
        a=ClassifierTools.crossValidationWithStats(knn, data, data.numInstances());
        System.out.println(" KNN LOOCV accuracy ="+a[0][0]);
        NaiveBayes nb = new NaiveBayes();
        a=ClassifierTools.crossValidationWithStats(nb, data, data.numInstances());
        System.out.println(" Naive Bayes LOOCV accuracy ="+a[0][0]);
        
 /*       try {
            tree.buildClassifier(data);
        System.out.println(" Tree ="+tree);
        Classifier cls = new J48();
         Evaluation eval = new Evaluation(data);
         Random rand = new Random(1);  // using seed = 1
         int folds = data.numInstances();
         eval.crossValidateModel(cls, data, folds, rand);
         System.out.println(eval.toSummaryString());        
        
        tree.getTechnicalInformation();
        } catch (Exception ex) {
            Logger.getLogger(GoodHonoursPrediction.class.getName()).log(Level.SEVERE, null, ex);
        }
        */
        
    }
    
}
