package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;


/**
 * Uses the Matthews Correlation Coefficient (MCC) to define the weighting of a module
 * MCC is a score calculated from the confusion matrix of the module's predictions
 * 
 * @author James Large
 */
public class MCCWeighting extends ModuleWeightingScheme {
    
    public MCCWeighting() {
        uniformWeighting = true;
    }
    
    @Override
    public double[] defineWeighting(ModulePredictions trainPredictions, int numClasses) {
        return makeUniformWeighting(computeMCC(trainPredictions.confusionMatrix), numClasses);
    }
    
    /**
     * todo could easily be optimised further
     */
    public double computeMCC(double[][] confusionMatrix) {
        
        double num=0.0;
        for (int k = 0; k < confusionMatrix.length; ++k)
            for (int l = 0; l < confusionMatrix.length; ++l)
                for (int m = 0; m < confusionMatrix.length; ++m) 
                    num += (confusionMatrix[k][k]*confusionMatrix[m][l])-
                            (confusionMatrix[l][k]*confusionMatrix[k][m]);

        double den1 = 0.0; 
        double den2 = 0.0;
        for (int k = 0; k < confusionMatrix.length; ++k) {
            
            double den1Part1=0.0;
            double den2Part1=0.0;
            for (int l = 0; l < confusionMatrix.length; ++l) {
                den1Part1 += confusionMatrix[l][k];
                den2Part1 += confusionMatrix[k][l];
            }

            double den1Part2=0.0;
            double den2Part2=0.0;
            for (int kp = 0; kp < confusionMatrix.length; ++kp)
                if (kp!=k) {
                    for (int lp = 0; lp < confusionMatrix.length; ++lp) {
                        den1Part2 += confusionMatrix[lp][kp];
                        den2Part2 += confusionMatrix[kp][lp];
                    }
                }
                      
            den1 += den1Part1 * den1Part2;
            den2 += den2Part1 * den2Part2;
        }
        
        return num / (Math.sqrt(den1)*Math.sqrt(den2));
    }
    
    /**
     * unoptimised version
     */
//    public double computeMCC(double[][] confusionMatrix) {
//        
//        double num=0.0;
//        for (int k = 0; k < confusionMatrix.length; ++k) {
//            for (int l = 0; l < confusionMatrix.length; ++l) {
//                for (int m = 0; m < confusionMatrix.length; ++m) {
//                  num += (confusionMatrix[k][k]*confusionMatrix[m][l])-
//                                (confusionMatrix[l][k]*confusionMatrix[k][m]);
//                }
//            }
//        }
//
//        double den1=0; 
//        for (int k = 0; k < confusionMatrix.length; ++k) {
//            
//            double  den1Part1=0;
//            for (int l = 0; l < confusionMatrix.length; ++l) 
//                den1Part1 += confusionMatrix[l][k];
//
//            double den1Part2=0;
//            for (int kp = 0; kp < confusionMatrix.length; ++kp)
//                if (kp != k)
//                    for (int lp = 0; lp < confusionMatrix.length; ++lp) 
//                      den1Part2 += confusionMatrix[lp][kp];
//                      
//            den1 += den1Part1*den1Part2;
//        }
//
//        double den2 = 0;
//        for (int k = 0; k < confusionMatrix.length; ++k){
//            
//            double  den2Part1=0;
//            for (int l = 0; l < confusionMatrix.length; ++l)
//                den2Part1 += confusionMatrix[k][l];
//
//            double den2Part2=0;
//            for (int kp = 0; kp < confusionMatrix.length; ++kp)
//                if (kp != k)
//                    for (int lp = 0; lp < confusionMatrix.length; ++lp)
//                        den2Part2 += confusionMatrix[kp][lp];
//            
//            den2 += den2Part1 * den2Part2;
//        }
//
//        return num / (Math.sqrt(den1)*Math.sqrt(den2));
//    }
}
