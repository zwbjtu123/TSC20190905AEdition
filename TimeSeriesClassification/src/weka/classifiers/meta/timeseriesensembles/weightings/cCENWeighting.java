/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.meta.timeseriesensembles.weightings;

import weka.classifiers.meta.timeseriesensembles.ModulePredictions;

/**
 *
 * @author James
 */
public class cCENWeighting extends CENWeighting {
    
    public cCENWeighting() {
        uniformWeighting = false;
    }
    
    @Override
    public double[] defineWeighting(ModulePredictions trainPredictions, int numClasses) {
       double[] weights = new double[numClasses];
        for (int j = 0; j < numClasses; j++) 
            weights[j] = cen_j(trainPredictions.confusionMatrix, j, numClasses);

        return weights;
    }
    
}
