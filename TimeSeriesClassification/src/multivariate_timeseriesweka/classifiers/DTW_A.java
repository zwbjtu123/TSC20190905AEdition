/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.classifiers;

import multivariate_timeseriesweka.elasticmeasures.DTW_D;
import multivariate_timeseriesweka.elasticmeasures.DTW_I;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;
import static timeseriesweka.filters.shapelet_transforms.quality_measures.InformationGain.entropy;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;
import utilities.generic_storage.Pair;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
 *
 * @author ABostrom
 */
public class DTW_A extends kNN{

    Instances train;
    
    double threshold;
    
    kNN DTW_I;
    kNN DTW_D;
    
    DistanceFunction I;
    DistanceFunction D;
    
    public DTW_A(int k){
        super(k);
        DTW_I = new kNN(1);
        DTW_D = new kNN(1);
        I = new DTW_I();
        D = new DTW_D();
    }
    
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        return result;
    }
    

    @Override
    public void buildClassifier(Instances data){
        train = data;
        
        threshold = learnThreshold(train);
        System.out.println("threshold = " + threshold);
        //build DTW_A. doesn't matter what function it uses for building as its' lazy.
        //default A to support a distance function of some kind.
        super.buildClassifier(data);
    }
     
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        setDistanceFunction(selectDistance(instance));
        return super.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        setDistanceFunction(selectDistance(instance));
        return super.distributionForInstance(instance);
    }
    
    double learnThreshold(Instances data){
        Pair<List<Double>, List<Double>> scores = findScores(data);
        List<Double> S_dSuccess = scores.var1;
        List<Double> S_iSuccess = scores.var2;
        
        double output;
        if(S_iSuccess.isEmpty() && S_dSuccess.isEmpty())
            output= 1;
        else if(!S_iSuccess.isEmpty() && S_dSuccess.isEmpty())
            output = Collections.max(S_iSuccess);
        else if(S_iSuccess.isEmpty() && !S_dSuccess.isEmpty())
            output = Collections.max(S_dSuccess);
        else
            //not sure this is exactly the same as the paper.
            //does use pseudo information gain to establish best ordering.
            output = calculateIGForThresholds(S_dSuccess, S_iSuccess); 
            
        return output;
    }
    
    double calculateIGForThresholds(List<Double> s_d, List<Double> s_i){
        
        //construct an orderline. we'll say d is class0, and i is class1.
        List<OrderLineObj> orderline = new ArrayList();
        for(Double s : s_d){
            orderline.add(new OrderLineObj(s, 0));
        }
        
        for(Double s : s_i){
            orderline.add(new OrderLineObj(s, 1));
        }
            
        return calculateThreshold(orderline);
    }
    
    public double calculateThreshold(List<OrderLineObj> orderline)
    {
        Collections.sort(orderline);

        // initialise class counts
        ClassDistribution lessClasses = new TreeSetClassDistribution();
        ClassDistribution greaterClasses = new TreeSetClassDistribution();

        for (OrderLineObj obj : orderline)
        {
            lessClasses.put(obj.getClassVal(), 0);
            greaterClasses.addTo(obj.getClassVal(), 1);
        }
        int countOfLeftClasses = 0;
        int countOfRightClasses = orderline.size();

        double lastDist = -1;
        double thisDist = -1;
        double bsfGain = Double.NEGATIVE_INFINITY;
        
        double bsf_threshold = 1;
        for (OrderLineObj ol : orderline)
        {
            thisDist = ol.getDistance();
            
            lessClasses.addTo(ol.getClassVal(), 1);
            greaterClasses.addTo(ol.getClassVal(), -1);

            countOfLeftClasses++;
            countOfRightClasses--;
            
            if (thisDist != lastDist)
            {
                // calculate the info gain below the threshold
                double lessFrac = (double) countOfLeftClasses / (double) orderline.size();
                double entropyLess = entropy(lessClasses);

                // calculate the info gain above the threshold
                double greaterFrac = (double) countOfRightClasses / (double) orderline.size();
                double entropyGreater = entropy(greaterClasses);

                //don't have a parent entropy, so just use 1. 
                //think its fine as long as its the same for all.
                double gain = 1.0 - lessFrac * entropyLess - greaterFrac * entropyGreater;
                if (gain > bsfGain)
                {
                    bsfGain = gain;
                    bsf_threshold = ol.getDistance();
                }
            }
            lastDist = thisDist;
        }
        
        return bsf_threshold;
    }
    
    Pair<List<Double>, List<Double>> findScores(Instances data){
        List<Double> S_dSuccess = new ArrayList<>();
        List<Double> S_iSuccess = new ArrayList<>();
        
        LinearNNSearch nn_I, nn_D;
        for(int i=0; i<data.numInstances(); i++){
            try {
                //LOOCV search for distances.
                Instances cv_train = data.trainCV(data.numInstances(), i);
                Instances cv_test = data.testCV(data.numInstances(), i);
                Instance test = cv_test.firstInstance();
                
                //setup our NN searches.
                nn_D = new LinearNNSearch(cv_train);
                nn_D.setDistanceFunction(D);
                nn_I = new LinearNNSearch(cv_train);
                nn_I.setDistanceFunction(I);
                
                //we know we only have one instance.
                double pred_d = nn_D.nearestNeighbour(test).classValue();
                double pred_i = nn_I.nearestNeighbour(test).classValue();
                double dist_d = nn_D.getDistances()[0];
                double dist_i = nn_I.getDistances()[0];
                double S = dist_d / dist_i;
                
                //if d is correct and i is incorrect.
                if(test.classValue() == pred_d && test.classValue() != pred_i)
                    S_dSuccess.add(S);
                //if d is incorrect and i is correct.
                if(test.classValue() != pred_d && test.classValue() == pred_i)
                    S_iSuccess.add(S);
            } catch (Exception ex) {
                System.out.println(ex);
            }
            
        }
       
        return new Pair(S_dSuccess, S_iSuccess);
    }

    double calculateS(Instances data, Instance instance){
        double minD = findMinDistance(data, instance, D);
        double minI = findMinDistance(data, instance, I);
        return minD / minI;
    }

    
    DistanceFunction selectDistance(Instance instance){
       double S = calculateS(train, instance);
       return S > threshold ? I : D;
    }
    
    double findMinDistance(Instances data, Instance inst, DistanceFunction dist){
        double min = dist.distance(data.get(0), inst);
        
        for (int i = 1; i < data.numInstances(); i++) {
            double temp = dist.distance(data.get(i), inst);
            if(temp < min)
                min = temp;
        }
        
        return min;
    }

}
