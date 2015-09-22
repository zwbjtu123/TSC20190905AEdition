package development.Jay;

import java.io.File;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.DTW;
import weka.core.elastic_distance_measures.DTW_DistanceBasic;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class GoreckiDerivativesEuclideanDistance extends EuclideanDistance{
    
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
    public static final String[] GORECKI_DATASETS = {
        //<editor-fold defaultstate="collapsed" desc="Datasets from the paper">
        "fiftywords", // 450,455,270,50
        "Adiac", // 390,391,176,37
        "Beef", // 30,30,470,5
        "CBF", // 30,900,128,3
        "Coffee", // 28,28,286,2
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "fish", // 175,175,463,7
        "GunPoint", // 50,150,150,2
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
        "SwedishLeaf", // 500,625,128,15
        "SyntheticControl", // 300,300,60,6
        "Trace", // 100,100,275,4
        "TwoPatterns", // 1000,4000,128,4
        "wafer", // 1000,6164,152,2
        "yoga"// 300,3000,426,2
        //</editor-fold>
    };
        
    protected double alpha;
    protected double a;
    protected double b;

    
    public GoreckiDerivativesEuclideanDistance(){
        this.a = 1;
        this.b = 0;
        this.alpha = -1;
        // defaults to no derivative input
    }
    public GoreckiDerivativesEuclideanDistance(Instances train){
        // this is what the paper suggests they use, but doesn't reproduce results. 
        //this.crossValidateForAlpha(train);
        
        // when cv'ing for a = 0:0.01:1 and b = 1:-0.01:0 results can be reproduced though, so use that
        this.crossValidateForAandB(train);
    }
    public GoreckiDerivativesEuclideanDistance(double alpha){
        this.alpha = alpha;
        this.a = Math.cos(alpha);
        this.b = Math.sin(alpha);
    }
    
    public GoreckiDerivativesEuclideanDistance(double a, double b){
        this.alpha = alpha;
        this.a = Math.cos(alpha);
        this.b = Math.sin(alpha);
    }

    @Override
    public double distance(Instance one, Instance two){
        return this.distance(one, two, Double.MAX_VALUE);
    }

    @Override
    public double distance(Instance one, Instance two, double cutoff, PerformanceStats stats){
        return this.distance(one,two,cutoff);
    }

    @Override
    public double distance(Instance first, Instance second, double cutoff){
        double dist = 0;
        double dirDist = 0;

        int classPenalty = 0;
        if(first.classIndex()>0){
            classPenalty=1;
        }

        double firstDir, secondDir;
        for(int i = 0; i < first.numAttributes()-classPenalty; i++){
            dist+= ((first.value(i)-second.value(i))*(first.value(i)-second.value(i)));

            // one less for derivatives, since we don't want to include the class value!
            // could skip the first instead of last, but this makes more sense for earlier early abandon
            if(i < first.numAttributes()-classPenalty-1){
                firstDir = first.value(i+1)-first.value(i);
                secondDir = second.value(i+1)-second.value(i);
                dirDist+= ((firstDir-secondDir)*(firstDir-secondDir));
            }

        }
        return(a*Math.sqrt(dist)+b*Math.sqrt(dirDist));
    }

    public double[] getNonScaledDistances(Instance first, Instance second){
        double dist = 0;
        double dirDist = 0;

        int classPenalty = 0;
        if(first.classIndex()>0){
            classPenalty=1;
        }

        double firstDir, secondDir;

        for(int i = 0; i < first.numAttributes()-classPenalty; i++){
            dist+= ((first.value(i)-second.value(i))*(first.value(i)-second.value(i)));

            if(i < first.numAttributes()-classPenalty-1){
                firstDir = first.value(i+1)-first.value(i);
                secondDir = second.value(i+1)-second.value(i);
                dirDist+= ((firstDir-secondDir)*(firstDir-secondDir));
            }
        }
        return new double[]{Math.sqrt(dist),Math.sqrt(dirDist)};
    }

    // implemented to mirror original MATLAB implementeation that's described in the paper (with appropriate modifications)
    public double crossValidateForAlpha(Instances train){
        double[] labels = new double[train.numInstances()];
        for(int i = 0; i < train.numInstances(); i++){
            labels[i] = train.instance(i).classValue();
        }

        double[] a = new double[ALPHAS.length];
        double[] b = new double[ALPHAS.length];

        for(int alphaId = 0; alphaId < ALPHAS.length; alphaId++){
            a[alphaId] = Math.cos(ALPHAS[alphaId]);
            b[alphaId] = Math.sin(ALPHAS[alphaId]);
        }

        int n = train.numInstances();
        int k = ALPHAS.length;
        int[] mistakes = new int[k];
//
//            // need to get the derivatives (MATLAB code uses internal diff function instead)
//            Instances dTrain = new GoreckiDerivativesDistance.GoreckiDerivativeFilter().process(train);

        double[] D;
        double[] L;
        double[] d;
        double dist;
        double dDist;

        double[] individualDistances;

        for(int i = 0; i < n; i++){

            D = new double[k];
            L = new double[k];
            for(int j = 0; j < k; j++){
                D[j]=Double.MAX_VALUE;
            }

            for(int j = 0; j < n; j++){
                if(i==j){
                    continue;
                }

                individualDistances = this.getNonScaledDistances(train.instance(i), train.instance(j));
                // have to be a bit different here, since we can't vectorise in Java
//                    dist = distanceFunction.distance(train.instance(i), train.instance(j));
//                    dDist = distanceFunction.distance(dTrain.instance(i), dTrain.instance(j));
                dist = individualDistances[0];
                dDist = individualDistances[1];

                d = new double[k];

                for(int alphaId = 0; alphaId < k; alphaId++){
                    d[alphaId] = a[alphaId]*dist+b[alphaId]*dDist;
                    if(d[alphaId] < D[alphaId]){
                        D[alphaId]=d[alphaId];
                        L[alphaId]=labels[j];
                    }
                }
            }

            for(int alphaId = 0; alphaId < k; alphaId++){
                if(L[alphaId]!=labels[i]){
                    mistakes[alphaId]++;
                }
            }
        }

        int bsfMistakes = Integer.MAX_VALUE;
        int bsfAlphaId = -1;
        for(int alpha = 0; alpha < k; alpha++){
            if(mistakes[alpha] < bsfMistakes){
                bsfMistakes = mistakes[alpha];
                bsfAlphaId = alpha;
            }
        }
        this.alpha = ALPHAS[bsfAlphaId];
        this.a = Math.cos(this.alpha);
        this.b = Math.sin(this.alpha);
//        System.out.println("bestAlphaId,"+bsfAlphaId);
        return (double)(train.numInstances()-bsfMistakes)/train.numInstances();
    }
    
    // changed to now return the predictions of the best alpha parameter
    public double[] crossValidateForAandB(Instances train){
        double[] labels = new double[train.numInstances()];
        for(int i = 0; i < train.numInstances(); i++){
            labels[i] = train.instance(i).classValue();
        }

        double[] a = new double[101];
        double[] b = new double[101];

        for(int alphaId = 0; alphaId <= 100; alphaId++){
            a[alphaId] = (double)(100-alphaId)/100;
            b[alphaId] = (double)alphaId/100;
        }

        int n = train.numInstances();
        int k = a.length;
        int[] mistakes = new int[k];

//            // need to get the derivatives (MATLAB code uses internal diff function instead)
//            Instances dTrain = new GoreckiDerivativesDistance.GoreckiDerivativeFilter().process(train);

        double[] D;
        double[] L;
        double[] d;
        double dist;
        double dDist;
        
        double[][] LforAll = new double[n][];

        double[] individualDistances;

        for(int i = 0; i < n; i++){
            
            D = new double[k];
            L = new double[k];
            for(int j = 0; j < k; j++){
                D[j]=Double.MAX_VALUE;
            }

            for(int j = 0; j < n; j++){
                if(i==j){
                    continue;
                }

                individualDistances = this.getNonScaledDistances(train.instance(i), train.instance(j));
                // have to be a bit different here, since we can't vectorise in Java
//                    dist = distanceFunction.distance(train.instance(i), train.instance(j));
//                    dDist = distanceFunction.distance(dTrain.instance(i), dTrain.instance(j));
                dist = individualDistances[0];
                dDist = individualDistances[1];

                d = new double[k];

                for(int alphaId = 0; alphaId < k; alphaId++){
                    d[alphaId] = a[alphaId]*dist+b[alphaId]*dDist;
                    if(d[alphaId] < D[alphaId]){
                        D[alphaId]=d[alphaId];
                        L[alphaId]=labels[j];
                    }
                }
            }

            for(int alphaId = 0; alphaId < k; alphaId++){
                if(L[alphaId]!=labels[i]){
                    mistakes[alphaId]++;
                }
            }
            LforAll[i] = L;
        }

        int bsfMistakes = Integer.MAX_VALUE;
        int bsfAlphaId = -1;
        for(int alpha = 0; alpha < k; alpha++){
            if(mistakes[alpha] < bsfMistakes){
                bsfMistakes = mistakes[alpha];
                bsfAlphaId = alpha;
            }
        }
        
        this.alpha = -1;
        this.a = a[bsfAlphaId];
        this.b = b[bsfAlphaId];
        double[] bestAlphaPredictions = new double[train.numInstances()];
        for(int i = 0; i < bestAlphaPredictions.length; i++){
            bestAlphaPredictions[i] = LforAll[i][bsfAlphaId];
        }
        return bestAlphaPredictions;
    }

    public double getA() {
        return a;
    }

    public double getB() {
        return b;
    }
    
    
    protected static int getCorrect(kNN knn, Instances train, Instances test) throws Exception{
        knn.buildClassifier(train);
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){
            if(test.instance(i).classValue()==knn.classifyInstance(test.instance(i))){
                correct++;
            }
        }
        return correct;
    }

    
    
    protected static class GoreckiEuclideanDistance extends EuclideanDistance{
        
        public GoreckiEuclideanDistance(){
            this.m_DontNormalize = true;
        }
        
        public double distance(Instance first, Instance second) {
            return distance(first, second, Double.MAX_VALUE);
        }
        
        public double distance(Instance first, Instance second, double cutoff){
            double dist = 0;
            int classPenalty = 0;
            if(first.classIndex()>0){
                classPenalty=1;
            }
            
            for(int i = 0; i < first.numAttributes()-classPenalty; i++){
                dist+= (first.value(i)-second.value(i))*(first.value(i)-second.value(i));
//                if(dist > cutoff){
//                    return Double.MAX_VALUE;
//                }
            }
           
            return Math.sqrt(dist);
        }
        
    }
    
    
    
    public static void main(String[] args) throws Exception{
        GoreckiDerivativesDTW.recreateGoreckiTable();
    }
    
}
