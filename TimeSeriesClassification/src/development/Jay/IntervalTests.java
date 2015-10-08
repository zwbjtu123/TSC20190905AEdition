

package development.Jay;


import development.IntervalBasedClassification;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.EuclideanDistance;
/**
 *
 * @author jay
 */
public class IntervalTests{
    
    static boolean classifierTest(double acc1, double acc2, int numInstances){

        double p=(acc1+acc2)/2; //Pooled proportion
        double se=Math.sqrt(p*(1-p)/(2*numInstances));     //Standard error
        double z=(acc1-acc2)/se; 
        if(z>1.96) return true;
        return false;
            
    }
    
    
    public static enum DecisionType{BEST_LONGEST, BEST_SHORTEST, ZBEST_LONGEST, ZBEST_SHORTEST};
    
    // proof of concept, naive implementation (could reuse calculations etc to make faster, to do later if it works)
    // no test of significance between intervals; just keep the best one
    public static void singleInterval(Instances inputTrain, Instances inputTest, Classifier classifier, int minInterval, int maxInterval, int intervalStep, int numFolds, DecisionType decisionType)  throws Exception{
        
        Instances fullTrain = new Instances(inputTrain);
        fullTrain.stratify(numFolds); // don't need to stratify the test data
        
        
        int fullLength = fullTrain.numAttributes()-1;
        int numInstances = fullTrain.numInstances();
        
        int correct;
        int bsfCorrect = -1;
        int bsfIntervalStart = -1;
        int bsfIntervalLength = -1;
        
        //for every interval size
        for(int intervalLength = maxInterval; intervalLength >= minInterval; intervalLength-=intervalStep){
            System.out.println("IntevalLength: "+intervalLength);
            // within length, for each start point
            for(int start = 0; start <= fullLength-intervalLength; start++){
                
                correct = evaluateInterval(fullTrain, start, intervalLength, fullLength, numFolds, classifier);
//                System.out.println("Start "+start+" Length "+intervalLength+" "+correct);
                // decision process
                switch(decisionType){
                    case BEST_LONGEST:
                        if(correct > bsfCorrect){
                            bsfCorrect = correct;
                            bsfIntervalStart = start;
                            bsfIntervalLength = intervalLength;
                        }
                        break;
                    case BEST_SHORTEST:
                        if(correct > bsfCorrect){
                            bsfCorrect = correct;
                            bsfIntervalStart = start;
                            bsfIntervalLength = intervalLength;
                        }else if(correct==bsfCorrect && intervalLength < bsfIntervalLength){
                            bsfCorrect = correct;
                            bsfIntervalStart = start;
                            bsfIntervalLength = intervalLength;
                        }
                        break;
                    case ZBEST_LONGEST:
                        if(correct > bsfCorrect && classifierTest((double)correct/numInstances, (double)bsfCorrect/numInstances, numInstances)){
                            System.out.println("was "+bsfIntervalStart+" "+bsfIntervalLength+" "+correct);
                            bsfCorrect = correct;
                            bsfIntervalStart = start;
                            bsfIntervalLength = intervalLength;
                            System.out.println("now "+bsfIntervalStart+" "+bsfIntervalLength+" "+correct);
                        }
                        break;
                    case ZBEST_SHORTEST:
                        boolean signifDiff = classifierTest((double)correct/numInstances, (double)bsfCorrect/numInstances, numInstances);
                        if(correct > bsfCorrect && signifDiff){
                            bsfCorrect = correct;
                            bsfIntervalStart = start;
                            bsfIntervalLength = intervalLength;
                        }else if(!signifDiff && intervalLength < bsfIntervalLength){
                            bsfCorrect = correct;
                            bsfIntervalStart = start;
                            bsfIntervalLength = intervalLength;
                        } 
                        break;
                }
                
            }
            
        }
        
        System.out.println("Best Interval: "+bsfCorrect+"("+(100.0/fullTrain.numInstances()*bsfCorrect)+")");
        System.out.println("Start:  "+bsfIntervalStart);
        System.out.println("Length: "+bsfIntervalLength);
        System.out.println();
        singleIntervalTest(inputTrain, inputTest, classifier, bsfIntervalStart, bsfIntervalLength);
        
    }
    
    private static int evaluateInterval(Instances fullTrain, int start, int intervalLength, int fullLength, int numFolds, Classifier classifier) throws Exception{
        int correct = 0;
        Instances cvTest;
        // create a copy of the data and chop it up
        Instances train = new Instances(fullTrain);
        for(int i = 0; i < start; i++){
            train.deleteAttributeAt(0);
        }
        for(int i = start+intervalLength; i < fullLength; i++){
            train.deleteAttributeAt(train.numAttributes()-2); //-2 because 0 index and class value\
        }


        for(int f = 0; f < numFolds; f++){
            classifier.buildClassifier(train.trainCV(numFolds, f));
            cvTest = train.testCV(numFolds, f);


            for(int i = 0; i < cvTest.numInstances(); i++){
                if(cvTest.instance(i).classValue()==classifier.classifyInstance(cvTest.instance(i))){
                    correct++;
                }
            }
        }
        return correct;
    }

    public static void singleIntervalTest(Instances inputTrain, Instances inputTest, Classifier classifier, int intervalStart, int intervalLength) throws Exception{
        Instances train = new Instances(inputTrain);
        Instances test = new Instances(inputTest);
        
        for(int i = 0; i < intervalStart; i++){
            train.deleteAttributeAt(0);
            test.deleteAttributeAt(0);
        }
        for(int i = intervalStart+intervalLength; i < train.numAttributes()-1; i++){
            train.deleteAttributeAt(train.numAttributes()-2); //-2 because 0 index and class value\
            test.deleteAttributeAt(train.numAttributes()-2); //-2 because 0 index and class value\
        }
        
        int correct = 0;
        classifier.buildClassifier(train);
        for(int i = 0; i < test.numInstances(); i++){
            if(test.instance(i).classValue()==classifier.classifyInstance(test.instance(i))){
                correct++;
            }
        }
        System.out.println("Test Accuracy:");
        System.out.println(correct+"/"+test.numInstances()+" ("+(100.0/test.numInstances()*correct)+")");
    
        
    }


    
    public static void main(String[] args) throws Exception{
        String dir = "/Users/jay/Dropbox/TSC Problems/";
//        String data= "ItalyPowerDemand";
//        String data= "SonyAIBORobotSurface";
        String data= "GunPoint";
        
        Instances train = utilities.ClassifierTools.loadData(dir+data+"/"+data+"_TRAIN");
        Instances test = utilities.ClassifierTools.loadData(dir+data+"/"+data+"_TEST");
        
        int numFolds = 10;
        kNN oneNN = new kNN();
        EuclideanDistance ed = new EuclideanDistance();
        ed.setDontNormalize(true);
        
        int minInterval = 3;
        int maxInterval = train.numAttributes()-1;
        int intervalStep = 1;
        
        singleInterval(train, test, oneNN, minInterval, maxInterval, intervalStep, numFolds, IntervalTests.DecisionType.ZBEST_LONGEST);
    }
    
}
