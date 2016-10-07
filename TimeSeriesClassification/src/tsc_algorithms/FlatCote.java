/**
 * NOTE: consider this code experimental. This is a first pass and may not be final; it has been informally tested but awaiting rigurous testing before being signed off.
 * Also note that file writing/reading from file is not currently supported (will be added soon)
 */

package tsc_algorithms;

import java.util.ArrayList;
import java.util.Random;
import tsc_algorithms.ElasticEnsemble;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.classifiers.meta.timeseriesensembles.depreciated.HESCA_05_10_16;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.PowerSpectrum;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class FlatCote extends AbstractClassifier{
    
    // Flat-COTE includes 35 constituent classifiers:
    //  -   11 from the Elastic Ensemble
    //  -   8 from the Shapelet Transform Ensemble
    //  -   8 from HESCA (ACF transformed)
    //  -   8 from HESCA (PS transformed)
    
    private enum EnsembleType{EE,ST,ACF,PS};
    private Instances train;
    
    
    private ElasticEnsemble ee;
    private HESCA st;
    private HESCA acf;
    private HESCA ps;
    
//    private ShapeletTransform shapeletTransform;
    private double[][] cvAccs;
    private double cvSum;
    
    private double[] weightByClass;
    
    @Override
    public void buildClassifier(Instances train) throws Exception{
        
        this.train = train;
        
        ee = new ElasticEnsemble();
        ee.buildClassifier(train);
        
        //ShapeletTransform shapeletTransform = ShapeletTransformFactory.createTransform(train);
        ShapeletTransform shapeletTransform = ShapeletTransformFactory.createTransformWithTimeLimit(train, 24); // now defaults to max of 24 hours
        shapeletTransform.supressOutput();
        st = new HESCA(shapeletTransform);
        st.buildClassifier(train);
        
        acf = new HESCA(new ACF());
        acf.buildClassifier(train);

        ps = new HESCA(new PowerSpectrum());
        ps.buildClassifier(train);
        
        cvAccs = new double[4][];
        cvAccs[0] = ee.getCVAccs();
        cvAccs[1] = st.getIndividualCvAccs();
        cvAccs[2] = acf.getIndividualCvAccs();
        cvAccs[3] = ps.getIndividualCvAccs();
        
        cvSum = 0;
        for(int e = 0; e < cvAccs.length;e++){
            for(int c = 0; c < cvAccs[e].length; c++){
                cvSum+=cvAccs[e][c];
            }
        }

    }
    
    @Override
    public double[] distributionForInstance(Instance test) throws Exception{
        weightByClass = null;
        classifyInstance(test);
        double[] dists = new double[weightByClass.length];
        for(int c = 0; c < weightByClass.length; c++){
            dists[c] = weightByClass[c]/this.cvSum;
        }
        return dists;
    }
    
    @Override
    public double classifyInstance(Instance test) throws Exception{
        
        double[][] preds = new double[4][];
        
        preds[0] = this.ee.classifyInstanceByConstituents(test);
        preds[1] = this.st.classifyInstanceByConstituents(test);
        preds[2] = this.acf.classifyInstanceByConstituents(test);
        preds[3] = this.ps.classifyInstanceByConstituents(test);
        
        weightByClass = new double[train.numClasses()];
        ArrayList<Double> bsfClassVals = new ArrayList<>();
        double bsfWeight = -1;
        
        for(int e = 0; e < preds.length; e++){
            for(int c = 0; c < preds[e].length; c++){
                weightByClass[(int)preds[e][c]]+=cvAccs[e][c];
//                System.out.print(preds[e][c]+",");
                if(weightByClass[(int)preds[e][c]] > bsfWeight){
                    bsfWeight = weightByClass[(int)preds[e][c]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(preds[e][c]);
                }else if(weightByClass[(int)preds[e][c]] > bsfWeight){
                    bsfClassVals.add(preds[e][c]);
                }
            }
        }
        
        if(bsfClassVals.size()>1){
            return bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
        }        
        return bsfClassVals.get(0);
    }
    
    public static void main(String[] args) throws Exception{
        
        FlatCote fc = new FlatCote();
        Instances train = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
        Instances test = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
        fc.buildClassifier(train);
        
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){
            if(fc.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
        }
        System.out.println("Acc");
        System.out.println(correct+"/"+test.numInstances());
        System.out.println((double)correct/test.numInstances());
        
                
    }
    
}