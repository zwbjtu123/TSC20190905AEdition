package development.Jay;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;
import weka.filters.timeseries.DerivativeFilter;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Implementation of 'Using Derivatives in Time Series Classification', Gorecki and Luczak
 * 
 */
public class GoreckiDerivativesDistance extends EuclideanDistance{

    public static final String DATA_DIR = "C:/Temp/Dropbox/TSC Problems/";
    public static final double[] ALPHAS = {
        //<editor-fold defaultstate="collapsed" desc="alpha values">
        1,
        1.01,
        1.02,
        1.03,
        1.04,
        1.05,
        1.06,
        1.07,
        1.08,
        1.09,
        1.1,
        1.11,
        1.12,
        1.13,
        1.14,
        1.15,
        1.16,
        1.17,
        1.18,
        1.19,
        1.2,
        1.21,
        1.22,
        1.23,
        1.24,
        1.25,
        1.26,
        1.27,
        1.28,
        1.29,
        1.3,
        1.31,
        1.32,
        1.33,
        1.34,
        1.35,
        1.36,
        1.37,
        1.38,
        1.39,
        1.4,
        1.41,
        1.42,
        1.43,
        1.44,
        1.45,
        1.46,
        1.47,
        1.48,
        1.49,
        1.5,
        1.51,
        1.52,
        1.53,
        1.54,
        1.55,
        1.56,
        1.57
//</editor-fold>
    };
    
    private double alpha;
    private double a;
    private double b;
    private EuclideanDistance distanceFunction;
    
    
    public GoreckiDerivativesDistance(double alpha, EuclideanDistance distanceFunction){
        this.alpha = alpha;
        this.a = Math.sin(alpha);
        this.b = Math.cos(alpha);
        this.distanceFunction = distanceFunction;
    }
    
    public GoreckiDerivativesDistance(double alpha){
        this.alpha = alpha;
        this.a = Math.sin(alpha);
        this.b = Math.cos(alpha);
        this.distanceFunction = new EuclideanDistance();
        this.distanceFunction.setDontNormalize(true);
    }
    
    @Override
    public double distance(Instance one, Instance two){
        return this.distance(one, two, Double.MAX_VALUE);
    }
    
    public double distance(Instance one, Instance two, double cutoff){
        
        // we can't guarentee that the EuclideanDistance object will have a distance(double[],double[]) method, so we have to stick with Instance objects
        Instance derOne = null;
        Instance derTwo = null;
        try{
            DerivativeFilter derFilter = new DerivativeFilter();
            Instances temp = new Instances(one.dataset(),0);
            temp.add(one);
            temp.add(two);
            temp = derFilter.process(temp);
            derOne = temp.instance(0);
            derTwo = temp.instance(1);
        }catch(Exception e){
            e.printStackTrace();
            return -1;
        }
        
        // now we have raw and (very hackily obtained) derivative instances.
        // next step, calculate the distances for each using specified distance function and apply weights
        double rawDist = this.distanceFunction.distance(one, two);
        double derDist = this.distanceFunction.distance(derOne, derTwo);
        
        return a*rawDist+b*derDist;
    }
    
    // keeping for safety, but updated below in the method below that returns the cv output. Delete when validated
    public static void prototype_crossValidateGorecki(String datasetName) throws Exception{
        // start with just ED to keep it simple
        
        
        Instances rawTrain = ClassifierTools.loadData(DATA_DIR+datasetName+"/"+datasetName+"_TRAIN");
        EuclideanDistance ed = new EuclideanDistance(rawTrain);
        ed.setDontNormalize(true);
        
        // use this later to make it generic
        EuclideanDistance df = ed;
        
        DerivativeFilter derivativeFilter = new DerivativeFilter();
        Instances derTrain = derivativeFilter.process(rawTrain);
        
        double[] classValues = new double[rawTrain.numInstances()];
        for(int i = 0; i < rawTrain.numInstances();i++){
            classValues[i] = rawTrain.instance(i).classValue();
        }
        
        double[] paramA = new double[ALPHAS.length];
        double[] paramB = new double[ALPHAS.length];
        
        for(int i = 0; i < ALPHAS.length; i++){
            paramA[i] = Math.cos(ALPHAS[i]);
            paramB[i] = Math.sin(ALPHAS[i]);
        }
        
        double[] bsfDistByAlpha;
        double[] bsfClassByAlpha;
        int[] correctByAlpha = new int[ALPHAS.length];
        
        boolean firstDist;
        double rawDist;
        double derDist;
        double thisDist;
        
        for(int testIns=0; testIns<rawTrain.numInstances();testIns++){
            firstDist = true;
            bsfDistByAlpha = new double[ALPHAS.length];
            bsfClassByAlpha = new double[ALPHAS.length];
            
            // can make this more efficient later
            for(int i = 0; i < bsfDistByAlpha.length; i++){
                bsfDistByAlpha[i] = Double.MAX_VALUE;//
                bsfClassByAlpha[i] = -1;//
            }
        
            // for each instance
            for(int trainIns = 0; trainIns < rawTrain.numInstances(); trainIns++){
                
                if(testIns==trainIns){
                    continue;
                }
                // get the standard distances
                rawDist = df.distance(rawTrain.instance(trainIns), rawTrain.instance(testIns));
                derDist = df.distance(derTrain.instance(trainIns), derTrain.instance(testIns));
                
                // now try all parameter possibilities, and update the best differece by alpha where applicable
                for(int p = 0; p < ALPHAS.length; p++){
                    thisDist = paramA[p]*df.distance(rawTrain.instance(trainIns), rawTrain.instance(testIns)) + paramB[p]*df.distance(derTrain.instance(trainIns), derTrain.instance(testIns));
                    if(thisDist < bsfDistByAlpha[p]){
                        bsfDistByAlpha[p] = thisDist;
                        bsfClassByAlpha[p] = rawTrain.instance(trainIns).classValue();
//                    }else if(thisDist == bsfDistByAlpha[p]){ // not handling this for now, can add in later if need be. For now, favours matches that happen first
                    }
                }
                
            }
            
            // So now we have a list of the nearest neighbours from the training data to the selected test ins. 
            // Now, compare the actual class val to the predictions, and update counts of correct.
            for(int p = 0; p < ALPHAS.length; p++){
                if(bsfClassByAlpha[p]==rawTrain.instance(testIns).classValue()){
                    correctByAlpha[p]++;
                }
            }
        }
        
        double thisAcc = -1;
        int bsfAlphaId = -1;
        double bsfAlphaAcc = -1;
        // now go through and find the best alpha
        for(int i = 0; i < correctByAlpha.length; i++){
            thisAcc = (double)correctByAlpha[i]/rawTrain.numInstances();
            System.out.println(ALPHAS[i]+"\t"+thisAcc);
            if(thisAcc > bsfAlphaAcc){
                bsfAlphaId = i;
                bsfAlphaAcc = thisAcc;
            }
        }
        
        System.out.println("Alpha:  "+ALPHAS[bsfAlphaId]);
        System.out.println("CV Acc: "+bsfAlphaAcc);
                
        
        
    }
    
    
    // returns an object that contains a GoreckiDistance object with the best cv params and the cv accuracy of the cv experiment 
    public static GoreckiCrossValidationOutput crossValidateGorecki(Instances train, EuclideanDistance internalDistanceFunction) throws Exception{
        // start with just ED to keep it simple
               
        Instances rawTrain = train;
        
        // use this later to make it generic
        EuclideanDistance df = internalDistanceFunction;
        
        DerivativeFilter derivativeFilter = new DerivativeFilter();
        Instances derTrain = derivativeFilter.process(rawTrain);
       
        double[] paramA = new double[ALPHAS.length];
        double[] paramB = new double[ALPHAS.length];
        
        for(int i = 0; i < ALPHAS.length; i++){
            paramA[i] = Math.cos(ALPHAS[i]);
            paramB[i] = Math.sin(ALPHAS[i]);
        }
        
        double[] bsfDistByAlpha;
        double[] bsfClassByAlpha;
        int[] correctByAlpha = new int[ALPHAS.length];
        
        boolean firstDist;
        double rawDist;
        double derDist;
        double thisDist;
        
        for(int testIns=0; testIns<rawTrain.numInstances();testIns++){
            firstDist = true;
            bsfDistByAlpha = new double[ALPHAS.length];
            bsfClassByAlpha = new double[ALPHAS.length];
            
            // can make this more efficient later, use a boolean to flag first distance or something
            for(int i = 0; i < bsfDistByAlpha.length; i++){
                bsfDistByAlpha[i] = Double.MAX_VALUE;//
                bsfClassByAlpha[i] = -1;//
            }
        
            // for each instance
            for(int trainIns = 0; trainIns < rawTrain.numInstances(); trainIns++){
                
                if(testIns==trainIns){
                    continue;
                }
                // get the standard distances
                rawDist = df.distance(rawTrain.instance(trainIns), rawTrain.instance(testIns));
                derDist = df.distance(derTrain.instance(trainIns), derTrain.instance(testIns));
                
                // now try all parameter possibilities, and update the best differece by alpha where applicable
                for(int p = 0; p < ALPHAS.length; p++){
                    thisDist = paramA[p]*df.distance(rawTrain.instance(trainIns), rawTrain.instance(testIns)) + paramB[p]*df.distance(derTrain.instance(trainIns), derTrain.instance(testIns));
                    if(thisDist < bsfDistByAlpha[p]){
                        bsfDistByAlpha[p] = thisDist;
                        bsfClassByAlpha[p] = rawTrain.instance(trainIns).classValue();
//                    }else if(thisDist == bsfDistByAlpha[p]){ // not handling this for now, can add in later if need be. For now, favours matches that happen first
                    }
                }
                
            }
            
            // So now we have a list of the nearest neighbours from the training data to the selected test ins. 
            // Now, compare the actual class val to the predictions, and update counts of correct.
            for(int p = 0; p < ALPHAS.length; p++){
                if(bsfClassByAlpha[p]==rawTrain.instance(testIns).classValue()){
                    correctByAlpha[p]++;
                }
            }
        }
        
        double thisAcc = -1;
        int bsfAlphaId = -1;
        double bsfAlphaAcc = -1;
        // now go through and find the best alpha
        for(int i = 0; i < correctByAlpha.length; i++){
            thisAcc = (double)correctByAlpha[i]/rawTrain.numInstances();
//            System.out.println(ALPHAS[i]+"\t"+thisAcc);
            if(thisAcc > bsfAlphaAcc){
                bsfAlphaId = i;
                bsfAlphaAcc = thisAcc;
            }
        }
        
//        System.out.println("Alpha:  "+ALPHAS[bsfAlphaId]);
//        System.out.println("CV Acc: "+bsfAlphaAcc);
                
        return new GoreckiCrossValidationOutput(new GoreckiDerivativesDistance(ALPHAS[bsfAlphaId], internalDistanceFunction),bsfAlphaAcc);
    }
    
    public static class GoreckiCrossValidationOutput{
        private GoreckiDerivativesDistance goreckiDistance;
        private double cvAccuracy;
        
        public GoreckiCrossValidationOutput(GoreckiDerivativesDistance goreckiDistance, double cvAccuracy){
            this.goreckiDistance = goreckiDistance;
            this.cvAccuracy = cvAccuracy;
        }

        public GoreckiDerivativesDistance getGoreckiDistance() {
            return goreckiDistance;
        }

        public double getCvAccuracy() {
            return cvAccuracy;
        }
        
    }
    
    public static void main(String[] args) throws Exception{
        
        // proof of concept
        
        /*  the main intuition behind the concept:
            - distance is based on a calculation between raw data and derivitives
            - these two distances are weighted with parameters a and b
            - params found through cv, but distances are consistent across all params
            - so the idea is calc distance once, then scan through params to find best match for each param
            - then once we have best dist for each k param, predict membership of the instance, calculate error for each param
            - at the end, pick the best one
        
        */
//        prototype_crossValidateGorecki("ItalyPowerDemand");
            
        String dataName = "ItalyPowerDemand";
        Instances train = ClassifierTools.loadData(DATA_DIR+dataName+"/"+dataName+"_TRAIN");

        EuclideanDistance ed = new EuclideanDistance(train);
        ed.setDontNormalize(true);
        GoreckiCrossValidationOutput cvOutput = crossValidateGorecki(train, ed);
        
        System.out.println("Best Alpha:  "+cvOutput.getGoreckiDistance().alpha);
        System.out.println("Best CV Acc: "+cvOutput.getCvAccuracy());
    }
}
