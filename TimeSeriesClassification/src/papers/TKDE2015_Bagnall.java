/*

Code to reproduce the results in the paper

Note that full results,including test accuracy of allthe components, can
be done with the method flatCOTE, which deconstructs the classifier.

To just run COTE, see the example in the method flatCOTE_Example. This uses
the COTE Classifier implementation in package 
weka.classifiers.meta.timeseriesensembles 

*/


package papers;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import statistics.simulators.ArmaModel;
import statistics.simulators.SimulateAR;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.timeseriesensembles.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.shapelet.QualityMeasures;
import weka.filters.Filter;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.ARMA;
import weka.filters.timeseries.PACF;
import weka.filters.timeseries.PowerSpectrum;
import weka.filters.timeseries.shapelet_transforms.*;
import weka.filters.unsupervised.attribute.RemoveUseless;

/**
Code to recreate results in paper XXX
* 
* Please note that these will be very slow. We actually distribute the 
* evaluation on a cluster, but have presented the methods as a whole for clarity. 
 */
public class TKDE2015_Bagnall {
/*Change these to where the data is and where you want the results. The format
we use is each problem is in its own directory, and there are a separate train
and test file. So ItalyPowerDemand is in the
    path+ItalyPowerDemand/ItalyPowerDemand_TEST.arff
    path+ItalyPowerDemand/ItalyPowerDemand_TEST.arff
    
    */ 
    
    static String path="C:/Users/ajb/Dropbox/TSC Problems/";
    static String resultsPath="C:/Users/ajb/Dropbox/Results/COTE/";

    
    /**
 * Generate simulated AR data sets and build an ensemble on the ACF, AR, PACF,
 * and combination of them all. This generates the results for Section XX
 * Figure XX
 * */
    public static void ACFSimulatedResults(){
        OutFile of=new OutFile(resultsPath+"ACF_Sim.csv");
        int minParas=1,maxParas=4;
        int nosCases=400;
        int[] nosCasesPerClass={200,200};
        int runs=30;
        WeightedEnsemble c;
        for(int seriesLength=100;seriesLength<=1000;seriesLength+=10){
            double[] mean=new double[4];
            double[]sd=new double[4];
            ArmaModel.setGlobalVariance(1);
            SimulateAR.setMinMaxPara(-0.1,0.1);
            try{
                for(int i=0;i<runs;i++){
                    //Generate data 
                    Instances all=SimulateAR.generateARDataSet(minParas,maxParas,seriesLength,nosCasesPerClass,true);
                    Instances train,test; 
                    all.randomize(new Random());

                    train=all;
                    test=new Instances(all);
                    for(int k=0;k<nosCases/2;k++){
                        train.delete(0);
                        test.delete(nosCases/2-1);
                    }
                    int maxLag=(train.numAttributes()-1)/4;
                    if(maxLag>100)
                        maxLag=100;
                    //1. ACF
                    ACF acf=new ACF();
                    acf.setMaxLag(maxLag);
                    acf.setNormalized(false);
                    Instances acfTrain=acf.process(train);
                    Instances acfTest=acf.process(test);
                    c=new WeightedEnsemble();
                    c.buildClassifier(acfTrain);
                    double a=ClassifierTools.accuracy(acfTest, c);
                    mean[0]+=a;
                    sd[0]+=a*a;
                    //2. ARMA 
                    ARMA arma=new ARMA();                        
                    arma.setMaxLag(maxLag);
                    arma.setUseAIC(false);
                    Instances arTrain=arma.process(train);
                    Instances arTest=arma.process(test);
                    c=new WeightedEnsemble();
                    c.buildClassifier(arTrain);
                    a=ClassifierTools.accuracy(arTest, c);
                    mean[1]+=a;
                    sd[1]+=a*a;
                    //3. PACF Full
                    //                   System.out.print(" PACF full ...");
                    PACF pacf=new PACF();
                    pacf.setMaxLag(maxLag);
                    Instances pacfTrain=pacf.process(train);
                    Instances pacfTest=pacf.process(test);
                    //                       System.out.println(" PACF num attributes="+pacfTrain.numAttributes());
                    c=new WeightedEnsemble();
                    c.buildClassifier(pacfTrain);
                    a=ClassifierTools.accuracy(pacfTest, c);
                    mean[2]+=a;
                    sd[2]+=a*a;                                              
                    //6. COMBO
                    Instances comboTrain=new Instances(acfTrain);
                    comboTrain.setClassIndex(-1);
                    comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
                    comboTrain=Instances.mergeInstances(comboTrain, pacfTrain);
                    comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
                    comboTrain=Instances.mergeInstances(comboTrain, arTrain);
                    comboTrain.setClassIndex(comboTrain.numAttributes()-1);
                    Instances comboTest=new Instances(acfTest);
                    comboTest.setClassIndex(-1);
                    comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
                    comboTest=Instances.mergeInstances(comboTest, pacfTest);
                    comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
                    comboTest=Instances.mergeInstances(comboTest, arTest);
                    comboTest.setClassIndex(comboTest.numAttributes()-1);
                    c= new WeightedEnsemble();
                    c.buildClassifier(acfTrain);
                    c.buildClassifier(comboTrain);
                    a=ClassifierTools.accuracy(comboTest, c);
                    mean[3]+=a;
                    sd[3]+=a*a;

                    }
//Calculate mean and SD
                    for(int i=0;i<sd.length;i++){
                        sd[i]=(sd[i]-mean[i]*mean[i]/runs)/runs;
                        mean[i]/=runs;
                        System.out.println(" \t "+mean[i]);
                    }
   //Write results to file
                    of.writeString(seriesLength+",");
                    for(int i=0;i<mean.length;i++){
                        of.writeString(mean[i]+",");
                    }
                    for(int i=0;i<mean.length;i++){
                        of.writeString(sd[i]+",");
                    }
                    of.writeString("\n");
               
           }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
           }
       }
    }

 /** 
  * Test the ACF transform on the UCR data sets  This generates the results for 
  * Section XX Figure XX   
  */
    public static void ACFAllProblemsResults(){
        OutFile of=new OutFile("ACF_AllProblems.csv");
        String path=""; //Add directory for problems
        int minParas=1,maxParas=4;
        int nosCases=400;
        int[] nosCasesPerClass={200,200};
        int runs=30;
        WeightedEnsemble c;
        double[] acc= new double[4];
        for(int fileNos=0;fileNos<DataSets.fileNames.length;fileNos++){
            double[] mean=new double[4];
            double[]sd=new double[4];
            ArmaModel.setGlobalVariance(1);
            SimulateAR.setMinMaxPara(-0.1,0.1);
            String fileName=DataSets.fileNames[fileNos];
            try{
                Instances test=utilities.ClassifierTools.loadData(path+fileName+"\\"+fileName+"_TEST");
                Instances train=utilities.ClassifierTools.loadData(path+fileName+"\\"+fileName+"_TRAIN");			
                int maxLag=(train.numAttributes()-1)/4;
                if(maxLag>100)
                    maxLag=100;
                //1. ACF
                ACF acf=new ACF();
                acf.setMaxLag(maxLag);
                acf.setNormalized(false);
                Instances acfTrain=acf.process(train);
                Instances acfTest=acf.process(test);
                c=new WeightedEnsemble();
                c.buildClassifier(acfTrain);
                acc[0]=ClassifierTools.accuracy(acfTest, c);
         //2. ARMA 
                ARMA arma=new ARMA();                        
                arma.setMaxLag(maxLag);
                arma.setUseAIC(false);
                Instances arTrain=arma.process(train);
                Instances arTest=arma.process(test);
                c=new WeightedEnsemble();
                c.buildClassifier(arTrain);
                acc[1]=ClassifierTools.accuracy(arTest, c);
          //3. PACF Full
                PACF pacf=new PACF();
                pacf.setMaxLag(maxLag);
                Instances pacfTrain=pacf.process(train);
                Instances pacfTest=pacf.process(test);
                //                       System.out.println(" PACF num attributes="+pacfTrain.numAttributes());
                c=new WeightedEnsemble();
                c.buildClassifier(pacfTrain);
                acc[2]=ClassifierTools.accuracy(pacfTest, c);
            //4. COMBO
                Instances comboTrain=new Instances(acfTrain);
                comboTrain.setClassIndex(-1);
                comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
                comboTrain=Instances.mergeInstances(comboTrain, pacfTrain);
                comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
                comboTrain=Instances.mergeInstances(comboTrain, arTrain);
                comboTrain.setClassIndex(comboTrain.numAttributes()-1);
                Instances comboTest=new Instances(acfTest);
                comboTest.setClassIndex(-1);
                comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
                comboTest=Instances.mergeInstances(comboTest, pacfTest);
                comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
                comboTest=Instances.mergeInstances(comboTest, arTest);
                comboTest.setClassIndex(comboTest.numAttributes()-1);
                c= new WeightedEnsemble();
                c.buildClassifier(comboTrain);
                acc[3]=ClassifierTools.accuracy(comboTest, c);
                 
   //Write results to file
                of.writeString(fileName+",");
                for(int i=0;i<acc.length;i++){
                    of.writeString(acc[i]+",");
                }
                of.writeString("\n");
           }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
           }
       }
    }

 /**
  * Using flat cote classifier. Note only the elastic ensemble is threaded in
  * this implementation
  * @param train
  * @param test
  * @param of
  * @return
  * @throws Exception 
  */
    public static double flatCOTE_Example(Instances train, Instances test, OutFile of) throws Exception{
        COTE c = new COTE();
        return ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);        
    }
 
/**
 * Deconstructed flat COTE classifier. 
 * @param train
 * @param test
 * @param of
 * @throws Exception 
 */    
    public static void flatCOTE(Instances train, Instances test, OutFile of) throws Exception{
//Form transforms
        //Change
        Instances changeTrain=ACF.formChangeCombo(train);
        Instances changeTest=ACF.formChangeCombo(test);
        System.out.println("Transformed change");
        //Power Spectrum
        PowerSpectrum ps=new PowerSpectrum();
        Instances psTrain=ps.process(train);
        Instances psTest=ps.process(test);
        System.out.println("Transformed PS");
        
        //Shapelet
        ShapeletTransformDistCaching s2=new ShapeletTransformDistCaching();
        
        s2.setNumberOfShapelets(train.numInstances()*10);
        s2.setShapeletMinAndMax(3, train.numAttributes()-1);
        s2.setDebug(false);
        s2.supressOutput();
        s2.turnOffLog();  
        s2.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.F_STAT);
        Instances shapeletTrain=s2.process(train);
        Instances shapeletTest=s2.process(test);
        System.out.println("Transformed Shapelet");
/*Remove any attributes that are all zero, because it knackers up
 The discretizer in BayesNet. Seems really unlikely        
        */
        RemoveUseless ru = new RemoveUseless();
        ru.setInputFormat(shapeletTrain);
        Instances sTrain2 = Filter.useFilter(shapeletTrain, ru);        
        Instances sTest2 = Filter.useFilter(shapeletTest, ru);        
 //Build all the classifiers       
//Elastic ensemble

        WeightedEnsemble weChange=new WeightedEnsemble();
        weChange.setWeightType(WeightedEnsemble.WeightType.PROPORTIONAL);
        weChange.buildClassifier(changeTrain);
        System.out.println("BUILT CHANGE");

        WeightedEnsemble wePS=new WeightedEnsemble();
        wePS.setWeightType(WeightedEnsemble.WeightType.PROPORTIONAL);
        wePS.buildClassifier(psTrain);
        System.out.println("BUILT PS");

        WeightedEnsemble weShapelet=new WeightedEnsemble();
        weShapelet.setWeightType(WeightedEnsemble.WeightType.PROPORTIONAL);
        weShapelet.buildClassifier(sTrain2);
        System.out.println("BUILT SHAPELET");
        
        ElasticEnsemble ee= new ElasticEnsemble();
        ee.turnAllClassifiersOn();
        ee.setEnsembleType(ElasticEnsemble.EnsembleType.Prop);
        ee.buildClassifier(train);
        System.out.println("BUILT EE");
        
//We need the training CV for all the ensembles. 
        double[] changeCVAccs=weChange.getCVAccs();
        double[] psCVAccs=wePS.getCVAccs();
        double[] shapeletCVAccs=weShapelet.getCVAccs();
        double[] elasticCVAccs=ee.getCVAccs();
        for(int j=0;j<elasticCVAccs.length;j++)
            elasticCVAccs[j]/=100.0;
        /*        for(double d:changeCVAccs)
            of.writeString(d+",");
        for(double d:psCVAccs)
            of.writeString(d+",");
        for(double d:shapeletCVAccs)
            of.writeString(d+",");
         of.writeString("\n");
*/
        double[] changeTestAcc=new double[changeCVAccs.length];
        double[] psTestAcc=new double[psCVAccs.length];
        double[] shapeletTestAcc=new double[shapeletCVAccs.length];
        double[] elasticTestAcc=new double[elasticCVAccs.length];
        double changeAcc=0,psAcc=0,shapeletAcc=0,elasticAcc=0,coteAcc=0;

        
        for(int i=0;i<test.numInstances();i++){
//Get predictions for all
            weChange.classifyInstance(changeTest.instance(i));
            wePS.classifyInstance(psTest.instance(i));
            weShapelet.classifyInstance(sTest2.instance(i));
            ee.classifyInstance(test.instance(i));
 //           if(i%10==0)
 //               System.out.println("CLASSIFIED EE instance "+i);
            int correctClass=(int)test.instance(i).classValue();
//            of.writeString(test.instance(i).classValue()+",");

//Recover component predictions
            double[] votes= new double[test.numClasses()];
            double[] allVotes= new double[test.numClasses()];
            double[] changePreds=weChange.getPredictions();            
//Form individual accuracies
            for(int j=0;j<changePreds.length;j++)
                if(correctClass==changePreds[j])
                    changeTestAcc[j]++;
//Form Change ensemble prediction            
            votes=new double[test.numClasses()];
            for(int j=0;j<changePreds.length;j++){
                votes[(int)changePreds[j]]+=changeCVAccs[j];
                allVotes[(int)changePreds[j]]+=changeCVAccs[j];
            }
//Find change prediction            
            int changeVote=0;
            for(int j=1;j<votes.length;j++){
                if(votes[changeVote]<votes[j])
                    changeVote=j;
            }
            
            double[] psPreds=wePS.getPredictions();
            for(int j=0;j<psPreds.length;j++)
                if(correctClass==psPreds[j])
                    psTestAcc[j]++;
//Form PS ensemble prediction            
            votes=new double[test.numClasses()];
            for(int j=0;j<psPreds.length;j++){
                votes[(int)psPreds[j]]+=psCVAccs[j];
                allVotes[(int)psPreds[j]]+=psCVAccs[j];
            }
//Find PS prediction            
            int psVote=0;
            for(int j=1;j<votes.length;j++){
                if(votes[psVote]<votes[j])
                    psVote=j;
            }
            
            double[] shapeletPreds=weShapelet.getPredictions();
            for(int j=0;j<shapeletPreds.length;j++)
                if(correctClass==shapeletPreds[j])
                    shapeletTestAcc[j]++;
//Form Shapelet ensemble prediction            
            votes=new double[test.numClasses()];
            for(int j=0;j<shapeletPreds.length;j++){
                votes[(int)shapeletPreds[j]]+=shapeletCVAccs[j];
                allVotes[(int)shapeletPreds[j]]+=shapeletCVAccs[j];
            }
//Find Shapelet prediction            
            int shapeletVote=0;
            for(int j=1;j<votes.length;j++){
                if(votes[shapeletVote]<votes[j])
                    shapeletVote=j;
            }
/*            for(double d:changePreds)
                of.writeString(d+",");
            for(double d:psPreds)
                of.writeString(d+",");
            for(double d:shapeletPreds)
                of.writeString(d+",");
*/
            double[] eePreds=ee.getPredictions();
            for(int j=0;j<eePreds.length;j++)
                if(correctClass==eePreds[j])
                    elasticTestAcc[j]++;
//Form Elastic ensemble prediction            
            votes=new double[test.numClasses()];
            for(int j=0;j<eePreds.length;j++){
                votes[(int)eePreds[j]]+=elasticCVAccs[j];
                allVotes[(int)eePreds[j]]+=elasticCVAccs[j];
            }
//Find Shapelet prediction            
            int elasticVote=0;
            for(int j=1;j<votes.length;j++){
                if(votes[elasticVote]<votes[j])
                    elasticVote=j;
            }
//Find COTE prediction
            int coteVote=0;
            for(int j=1;j<allVotes.length;j++){
                if(allVotes[coteVote]<allVotes[j])
                    coteVote=j;
            }

// Write all preds to file, including ensemble
//            if(i%10==0)
//                System.out.println("Finished case "+(i+1)+" of "+test.numInstances());
//            of.writeString(changeVote+","+psVote+","+shapeletVote+","+elasticVote+","+coteVote);
//            of.writeString("\n");
//Update ensemble accuracies
            if(changeVote==correctClass)
                changeAcc++;
            if(psVote==correctClass)
                psAcc++;
            if(shapeletVote==correctClass)
                shapeletAcc++;
            if(elasticVote==correctClass)
                elasticAcc++;
            if(coteVote==correctClass)
                coteAcc++;
        }
//        of.writeString("\n");
//Write the individual classifier accuracies to file        
        for(double d:changeTestAcc)
                of.writeString((d/test.numInstances())+",");
        for(double d:psTestAcc)
                of.writeString((d/test.numInstances())+",");
        for(double d:shapeletTestAcc)
                of.writeString((d/test.numInstances())+",");
        for(double d:elasticTestAcc)
                of.writeString((d/test.numInstances())+",");
        of.writeString(changeAcc/test.numInstances()+",");
        of.writeString(psAcc/test.numInstances()+",");
        of.writeString(shapeletAcc/test.numInstances()+",");
        of.writeString(elasticAcc/test.numInstances()+",");
        of.writeString(coteAcc/test.numInstances()+",");
        of.writeString("\n");
   
    }
    
 /**
  * Code to run a random resample problem. See figures XX and Section XX
  */ 
    public static int repNumber=1;
    public static void resampleExperiment(String problem, int reps) throws Exception{
        Instances train =ClassifierTools.loadData(path+problem+"/"+problem+"_TRAIN");
        Instances test =ClassifierTools.loadData(path+problem+"/"+problem+"_TEST");
        OutFile results= new OutFile(resultsPath+problem+repNumber+"Resample.csv");
        int testSize=test.numInstances();
        int trainSize=train.numInstances();
        System.out.println(" PROBLEM ="+problem+" Test size= "+testSize+" Train size ="+trainSize);
        if(problem.equals("Beef")||problem.equals("Coffee")||problem.equals("OliveOil")){
            NormalizeCase nc = new NormalizeCase();
            train=nc.process(train);
            test=nc.process(test);
        }
        Instances all = new Instances(train);
        all.addAll(test);
        for(int i=0;i<reps;i++){
            System.out.println(" Repetition "+(i+1)+" of "+reps);
            Random r= new Random();
            all.randomize(r);
            Instances tr = new Instances(all);
            Instances te = new Instances(all,0);
            for(int j=0;j<testSize;j++){
                Instance ins=tr.remove(0);
                te.add(ins);
            }
            if(tr.numInstances()!=trainSize || te.numInstances()!=testSize){
                System.out.println("ERROR: Mismatch sizes on reps trainSize "+trainSize+" split trainsize ="+tr.numInstances());
                System.out.println("ERROR: Mismatch sizes on reps testSize "+testSize+" split testsize ="+te.numInstances());
                System.exit(0);
            }
            flatCOTE(tr,te,results);
        }
    }
    
    
/**
 * 
 */    
    public static void testUCR(){
        //for(String s: DataSets.ucrNames)
        String s="FaceFour";
        
        {
            Instances train=ClassifierTools.loadData(path+s+"/"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(path+s+"/"+s+"_TEST");
            kNN knn = new kNN();
            knn.setDistanceFunction(new BasicDTW());
            double a = ClassifierTools.singleTrainTestSplitAccuracy(knn, train, test);
            System.out.println(s+","+a);
        }
        
    }
    
    
    public static void processClusterResults(){
     //Check which ones have any entries and count them   
        String path="C:\\Users\\ajb\\Dropbox\\Results\\COTE\\Complete\\";
        OutFile of=new OutFile(path+"TrainTest.csv");
        OutFile of2=new OutFile(path+"ResampleMeansandVar.csv");
        int present=0;
        int absent=0;
        int dtwPos=8+8+8+2;   //
       for(String s:DataSets.fileNames){
           if(InFile.fileExists(path+s+"Resample.csv")){
               InFile f=new InFile(path+s+"Resample.csv");
               int lines =f.countLines();
               if(lines==0){
                   f.closeFile();
                    System.out.println(path+s+"resample.csv is empty");
                    f=null;
                    InFile.deleteFile(path+s+"resample.csv");
               }else{
                    System.out.println(s+" has "+lines+" resamples");
                   present++;
                   f=new InFile(path+s+"Resample.csv");
                   String results=f.readLine();
                   of.writeLine(s+","+results);
                   of2.writeString(s+","+lines+",");
                   String[] data=results.split(",");
                   double[] sum=new double[data.length];
                   double[] sumSq=new double[data.length];
                   for(int i=0;i<sum.length;i++){
                       sum[i]=Double.parseDouble(data[i]);
                       sumSq[i]=sum[i]*sum[i];
                   }
                   double testResult=sum[sum.length-1];
                   for(int count=2;count<=lines;count++){
                       for(int i=0;i<sum.length;i++){
                           double t=f.readDouble();
                           sum[i]+=t;
                           sumSq[i]+=t*t;
                       }
                   }
                   for(int i=0;i<sum.length;i++){
                       sum[i]/=lines;
                       sumSq[i]=sumSq[i]/lines- sum[i]*sum[i];
                   }
                   of2.writeString(testResult+","+sum[sum.length-1]+","+sumSq[sum.length-1]+","+sum[dtwPos]+","+sumSq[dtwPos]+",,");
                   
                   
                   
                   for(int i=0;i<sum.length;i++)
                        of2.writeString(sum[i]+",");
                    of2.writeString(",,");
                   for(int i=0;i<sum.length;i++)
                        of2.writeString(sumSq[i]+",");
        of2.writeString("\n");
                        
                   
               }
               
           }
           else{
//               System.out.println(path+DataSets.fileNames[i]+"resample.csv does not exist");
               absent++;
           }
       } 
               System.out.println("PRESENT ="+present+" ABSENT = "+absent);
       
        
    }
    
    
    
    public static void main(String[] args){
 //       combine("Trace");
 //       combine("OliveOil");
 //      combine("FacesUCR");
        processClusterResults();
      System.exit(0);
 //       testUCR();    
//        String fileName="ItalyPowerDemand";
        String fileName="FacesUCR";        
            path="TSC Problems/";
            resultsPath="Results/FacesUCR/";
        if(args[0]!=null){//Cluster run
            repNumber=Integer.parseInt(args[0])-1;
//            fileName=DataSets.smallFileNames[index];
        }
    try {
        resampleExperiment(fileName,1);
        } catch (Exception ex) {
            Logger.getLogger(TKDE2015_Bagnall.class.getName()).log(Level.SEVERE, null, ex);
            System.exit(0);
        }
        
        
    }        
   public static void combine(String name){
       OutFile of = new OutFile("C:/Users/ajb/Dropbox/Results/COTE/"+name+"Resample.csv");
       for(int i=0;i<50;i++){
           InFile f = new InFile("C:/Users/ajb/Dropbox/Results/COTE/"+name+"/"+name+i+"Resample.csv");
           of.writeLine(f.readLine());
       }
       
   }     

}
