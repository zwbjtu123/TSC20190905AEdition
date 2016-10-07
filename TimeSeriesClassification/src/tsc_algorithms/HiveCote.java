/**
 * NOTE: consider this code experimental. This is a first pass and may not be final; it has been informally tested but awaiting rigurous testing before being signed off.
 * Also note that file writing/reading from file is not currently supported (will be added soon)
 */


package tsc_algorithms;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import tsc_algorithms.cote.HiveCoteModule;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class HiveCote extends AbstractClassifier{
    
    private ArrayList<Classifier> classifiers;
    private ArrayList<String> names;
    private ConstituentHiveEnsemble[] modules;
    private boolean verbose = false;
    private int maxCvFolds = 10;// note: this only affects manual CVs from this class using the crossvalidate method. This will not affect internal classifier cv's if they are set within those classes
    
    public HiveCote(Instances train){
        this.setDefaultEnsembles(train);
    }
    
    public HiveCote(ArrayList<Classifier> classifiers, ArrayList<String> classifierNames){
        this.classifiers = new ArrayList<>();
        this.names = new ArrayList<>();
        Collections.copy(classifiers, this.classifiers);
        Collections.copy(classifierNames, this.names);
    }

    private void setDefaultEnsembles(Instances train){
        
        classifiers = new ArrayList<>();
        names = new ArrayList<>();
        
        classifiers.add(new ElasticEnsemble());
//        ShapeletTransform shoutyThing = ShapeletTransformFactory.createTransform(train);
        ShapeletTransform shoutyThing = ShapeletTransformFactory.createTransformWithTimeLimit(train, 24); // changed to default to 24 hours max shapelet discovery
        shoutyThing.supressOutput();
        classifiers.add(new HESCA(shoutyThing));
        RISE rise = new RISE();
        rise.setTransformType(RISE.Filter.PS_ACF);
        classifiers.add(rise);
        classifiers.add(new BOSSEnsemble());
        classifiers.add(new TSF());
        
        names.add("EE");
        names.add("ST");
        names.add("RISE");
        names.add("BOSS");
        names.add("TSF");
    }
    
    @Override
    public void buildClassifier(Instances train) throws Exception{
        optionalOutputLine("Start of training");
                
        modules = new ConstituentHiveEnsemble[classifiers.size()];
        
        double ensembleAcc;
        for(int i = 0; i < classifiers.size(); i++){
            
            // if classifier is an implementation of HiveCoteModule, no need to cv for ensemble accuracy as it can self-report
            // e.g. of the default modules, EE, HESCA, and BOSS should all have this fucntionality (group a); RISE and TSF do not currently (group b) so must manualy cv
            if(classifiers.get(i) instanceof HiveCoteModule){
                optionalOutputLine("training (group a): "+this.names.get(i));
                classifiers.get(i).buildClassifier(train);
                modules[i] = new ConstituentHiveEnsemble(this.names.get(i), this.classifiers.get(i), ((HiveCoteModule) classifiers.get(i)).getEnsembleCvAcc());
            // else we must do a manual cross validation to get the module's encapsulated cv acc
            // note this isn't optimal; would be better to change constituent ensembles to self-record cv acc during training, rather than cv-ing and then building
            // however, this is effectively a wrapper so we can add any classifier to the collective without worrying about implementation support
            }else{
                optionalOutputLine("crossval (group b): "+this.names.get(i));
                ensembleAcc = crossValidate(classifiers.get(i), train, maxCvFolds);
                optionalOutputLine("training (group b): "+this.names.get(i));
                classifiers.get(i).buildClassifier(train);                
                modules[i] = new ConstituentHiveEnsemble(this.names.get(i), this.classifiers.get(i), ensembleAcc);
            }
            optionalOutputLine("done "+modules[i].classifierName);
        }        

        if(verbose){
            printModuleCvAccs();
        }
       
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        
        double[] hiveDists = new double[instance.numClasses()];
        double[] moduleDists;
        double moduleWeight;
        
        
        for(int m = 0; m < modules.length; m++){
            moduleDists = modules[m].classifier.distributionForInstance(instance);
            moduleWeight = modules[m].ensembleCvAcc;
            for(int c = 0; c < hiveDists.length; c++){
                hiveDists[c] += moduleDists[c]*moduleWeight;
            }
        }
        return hiveDists;
    }
    
    
    public double[] classifyInstanceByEnsemble(Instance instance) throws Exception{
        
        double[] output = new double[modules.length];
        
        for(int m = 0; m < modules.length; m++){
            output[m] = modules[m].classifier.classifyInstance(instance);
        }
        return output;
    }
    
    public void printModuleCvAccs() throws Exception{
        if(this.modules==null){
            throw new Exception("Error: modules don't exist. Train classifier first.");
        }
        System.out.println("CV accs by module:");
        System.out.println("------------------");
        StringBuilder line1 = new StringBuilder();
        StringBuilder line2 = new StringBuilder();
        for (ConstituentHiveEnsemble module : modules) {
            line1.append(module.classifierName).append(",");
            line2.append(module.ensembleCvAcc).append(",");
        }
        System.out.println(line1);
        System.out.println(line2);
        System.out.println();
    }
    
    private void makeShouty(){
        this.verbose = true;
    }
    
    private void optionalOutputLine(String message){
        if(this.verbose){
            System.out.println(message);
        }
    }
    
    public void setMaxCvFolds(int maxFolds){
        this.maxCvFolds = maxFolds;
    }
 
    public double crossValidate(Classifier classifier, Instances train, int maxFolds) throws Exception{
        
        int numFolds = maxFolds;
        if(numFolds <= 1 || numFolds > train.numInstances()){
            numFolds = train.numInstances();
        }

        Random r = new Random();

        ArrayList<Instances> folds = new ArrayList<>();
        ArrayList<ArrayList<Integer>> foldIndexing = new ArrayList<>();

        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(train,0));
            foldIndexing.add(new ArrayList<>());
        }

        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < train.numInstances(); i++){
            instanceIds.add(i);
        }
        Collections.shuffle(instanceIds, r);

        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < train.numClasses(); i++){
            byClass.add(new Instances(train,0));
            byClassIndices.add(new ArrayList<>());
        }

        int thisInstanceId;
        double thisClassVal;
        for(int i = 0; i < train.numInstances(); i++){
            thisInstanceId = instanceIds.get(i);
            thisClassVal = train.instance(thisInstanceId).classValue();

            byClass.get((int)thisClassVal).add(train.instance(thisInstanceId));
            byClassIndices.get((int)thisClassVal).add(thisInstanceId);
        }

         // now stratify        
        Instances strat = new Instances(train,0);
        ArrayList<Integer> stratIndices = new ArrayList<>();
        int stratCount = 0;
        int[] classCounters = new int[train.numClasses()];

        while(stratCount < train.numInstances()){

            for(int c = 0; c < train.numClasses(); c++){
                if(classCounters[c] < byClass.get(c).size()){
                    strat.add(byClass.get(c).instance(classCounters[c]));
                    stratIndices.add(byClassIndices.get(c).get(classCounters[c]));
                    classCounters[c]++;
                    stratCount++;
                }
            }
        }


        train = strat;
        instanceIds = stratIndices;

        double foldSize = (double)train.numInstances()/numFolds;

        double thisSum = 0;
        double lastSum = 0;
        int floor;
        int foldSum = 0;


        int currentStart = 0;
        for(int f = 0; f < numFolds; f++){


            thisSum = lastSum+foldSize+0.000000000001;  // to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
            floor = (int)thisSum;

            if(f==numFolds-1){
                floor = train.numInstances(); // to make sure all instances are allocated in case of double imprecision causing one to go missing
            }

            for(int i = currentStart; i < floor; i++){
                folds.get(f).add(train.instance(i));
                foldIndexing.get(f).add(instanceIds.get(i));
            }

            foldSum+=(floor-currentStart);
            currentStart = floor;
            lastSum = thisSum;
        }

        if(foldSum!=train.numInstances()){
            throw new Exception("Error! Some instances got lost file creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
        }


        Instances trainLoocv;
        Instances testLoocv;

        double pred, actual;
        double[] predictions = new double[train.numInstances()];

        int correct = 0;
        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake

        for(int testFold = 0; testFold < numFolds; testFold++){

            trainLoocv = null;
            testLoocv = new Instances(folds.get(testFold));

            for(int f = 0; f < numFolds; f++){
                if(f==testFold){
                    continue;
                }
                temp = new Instances(folds.get(f));
                if(trainLoocv==null){
                    trainLoocv = temp;
                }else{
                    trainLoocv.addAll(temp);
                }
            }

            classifier.buildClassifier(trainLoocv);
            for(int i = 0; i < testLoocv.numInstances(); i++){
                pred = classifier.classifyInstance(testLoocv.instance(i));
                actual = testLoocv.instance(i).classValue();
                predictions[foldIndexing.get(testFold).get(i)] = pred;
                if(pred==actual){
                    correct++;
                }
            }
        }

        return (double)correct/train.numInstances();
//        return predictions;
    }


    private class ConstituentHiveEnsemble{

        public final Classifier classifier;
        public final double ensembleCvAcc;
        public final String classifierName;

        public ConstituentHiveEnsemble(String classifierName, Classifier classifier, double ensembleCvAcc){
            this.classifierName = classifierName;
            this.classifier = classifier;
            this.ensembleCvAcc = ensembleCvAcc;
        }
    }
    
    public static void main(String[] args) throws Exception{
       
        String datasetName = "ItalyPowerDemand";
//        String datasetName = "MoteStrain";
        
        Instances train = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData("C:/users/sjx07ngu/dropbox/tsc problems/"+datasetName+"/"+datasetName+"_TEST");

        HiveCote hive = new HiveCote(train);
        hive.makeShouty();
        
        hive.setDefaultEnsembles(train);
        hive.buildClassifier(train);
        
        int correct = 0;
        double[] predByEnsemble;
        int[] correctByEnsemble = new int[hive.modules.length];
        for(int i = 0; i < test.numInstances(); i++){
            if(hive.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
            predByEnsemble = hive.classifyInstanceByEnsemble(test.instance(i)); // not efficient, just informative. can add this in to the classifyInstance in a hacky way later if need be
            for(int m = 0; m < predByEnsemble.length; m++){
                if(predByEnsemble[m]==test.instance(i).classValue()){
                    correctByEnsemble[m]++;
                }
            }
        }
        System.out.println("Overall Acc: "+(double)correct/test.numInstances());
        System.out.println("Acc by Module:");
    
        StringBuilder line1 = new StringBuilder();
        StringBuilder line2 = new StringBuilder();
        for(int m = 0; m < hive.modules.length; m++){
            line1.append(hive.modules[m].classifierName).append(",");
            line2.append((double)correctByEnsemble[m]/test.numInstances()).append(",");
        }
        System.out.println(line1);
        System.out.println(line2);
    }
    
}
