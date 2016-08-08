/*
CODE to generate and process the results for

The Heterogeneous Ensemble of Standard Classification Algorithms (HESCA)

Classifier accuracy on test data:



*/
package UCI_classification;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.TreeSet;
import new_COTE_experiments.BiasVarianceEvaluation;
import new_COTE_experiments.BiasVarianceEvaluation.BVResult;
import statistics.tests.OneSampleTests;
import weka.classifiers.meta.timeseriesensembles.SaveableEnsemble;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveCVAccuracy;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.meta.timeseriesensembles.HESCA;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
1. Form train/test mapping for resample
2. Filter out stupid problems (too small,too easy)
3. Set up to run all experiments with resamples
 * 
 */
public class UCIClassification {
    public static String[] uciFileNames={"abalone",
        "acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std",
        "balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc",
        "breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car",
        "cardiotocography-10clases","cardiotocography-3clases","chess-krvk","chess-krvkp",
        "congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        "connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram",
        "ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth",
        "heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley",
        "horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display",
        "lenses","letter","libras","low-res-spect","lung-cancer",
        //"lymphography",
        "magic","mammographic",
        "miniboone","molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom",
        "musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f",
        "oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks",
        "parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L",
        "pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning",
        "plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm",
        "seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit",
        "statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle",
        "statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe",
        "titanic",
        "trains",
        "twonorm","vertebral-column-2clases","vertebral-column-3clases",
        "wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white",
        "yeast","zoo"};

    static String problemPath="C:/UCI/UCIData/";
    static String resultsPath="C:/UCI/UCIResults/";
    public static String algorithm="WE";
    static double prop=0.3;
/************ Helper Methods called by the main processors*******************/
    public static Classifier setClassifier(String classifier){
        HESCA c=null;
        switch(classifier){
            case "HESCA": case "WE": default:
                c=new HESCA();
                c.setWeightType(HESCA.WeightType.PROPORTIONAL);
                c.setCVPath(problemPath);
                break;
//May differentiate classifiers later       //May differentiate classifiers later       
        }
        return c;
    }   
    public static void describeData(){
          OutFile of=new OutFile(problemPath+"DataDesription.csv");
          for(String a:uciFileNames){
              Instances data=ClassifierTools.loadData(problemPath+a+"/"+a);
              System.out.println(a+"  Nos Cases ="+data.numInstances()+" Nos Atts ="+(data.numAttributes()-1)+" Nos Classes ="+data.numClasses());
              of.writeLine(a+","+data.numInstances()+","+(data.numAttributes()-1)+","+data.numClasses());
          }
      }
    public static void findAllMappings(){
          for(String s:uciFileNames)
                findMappings(s);
          
      }
    public static void findMappings(String problem){
        Instances all=ClassifierTools.loadData(problemPath+problem+"/"+problem);
        all.insertAttributeAt(new Attribute("Index"), 0);
        int value=0;
        for(Instance in:all){
            in.setValue(0, value++);
        }
        OutFile folds=new OutFile(resultsPath+"Folds/"+problem+".csv");
        for(int i=0;i<100;i++){
           Instances[] split=InstanceTools.resampleInstances(all, i, 0.3);
           folds.writeLine("Fold,"+i);
            int count=0;
            for(Instance in:split[0]){
                if(in.value(0)<0.3*all.numInstances())
                    folds.writeLine(count+","+(int)in.value(0)+",TRAIN,TRAIN");
                else
                    folds.writeLine(count+","+(int)in.value(0)+",TEST,TRAIN");
                count++;
            }
            for(Instance in:split[1]){
                if(in.value(0)<0.3*all.numInstances())
                    folds.writeLine(count+","+(int)in.value(0)+",TRAIN,TEST");
                else
                    folds.writeLine(count+","+(int)in.value(0)+",TEST,TEST");
                count++;
            }
        }
    }    
/**
 * Perform a CV whilst 
 * @param c
 * @param allData
 * @param m
 * @return 
 */
		
    
    
/** PART 1: ******* GENERATE TEST AND TRAIN PREDICTIONS *******/    
    
/**
 * TEST: Generates the test predictions
 * @param args 
 *  args[0] = classifier name, classifier built in setClassifier
 *  args[1] = problem name, arff stored in problemPath+problem+"/"+problem+".arff"
 *  args[2] = fold number, determines the split into train test of the problem file
 * 
 * OutPut: 
 * Internal 
 * 
 * Test predictions for full ensemble problem/fold in
 *      resultsPath+classifier+"/Predictions/"+problem+"/testPreds"+fold+".csv";
 * Test predictions for all components in
 *      resultsPath+classifier+"/Predictions/"+problem+"/internalTestPreds_"+fold+".csv"
 * Train CV values for all components in
 *      resultsPath+classifier+"/Predictions/"+problem+"/internalCV_"+fold+".csv"
 * Train fold CV predictions for all components in 
 *      resultsPath+classifier+"/Predictions/"+problem+"/internalTrainPreds_"+fold+".csv"
              **/       
    public static void HESCATestAccuracyExperiment(String[] args){
  
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
//Set up file structure if necessary   
        File f=new File(resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String preds=resultsPath+classifier+"/Predictions";
        f=new File(preds);
        if(!f.exists())
            f.mkdir();
        preds=preds+"/"+problem;
        f=new File(preds);
        if(!f.exists())
            f.mkdir();

//Check whether fold already exists, if so, dont do it, just quit
        f=new File(preds+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
            Classifier c=setClassifier(classifier);
            Instances all=ClassifierTools.loadData(problemPath+problem+"/"+problem);
            Instances [] data=InstanceTools.resampleInstances(all, fold, prop);
            
            double acc=0;
            OutFile p=new OutFile(preds+"/testFold"+fold+".csv");
    // hack here to save internal CV for further ensembling   
            if(c instanceof SaveableEnsemble)
               ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+fold+".csv",preds+"/internalTestPreds_"+fold+".csv");
            try{              
                c.buildClassifier(data[0]);
                int[][] predictions=new int[data[1].numInstances()][2];
                for(int j=0;j<data[1].numInstances();j++)
                {
                    predictions[j][0]=(int)data[1].instance(j).classValue();
                    predictions[j][1]=(int)c.classifyInstance(data[1].instance(j));
                    if(predictions[j][0]==predictions[j][1])
                        acc++;
                }
                acc/=data[1].numInstances();
                String[] names=preds.split("/");
                p.writeLine(names[names.length-1]+","+c.getClass().getName()+",test");
                if(c instanceof SaveCVAccuracy)
                    p.writeLine(((SaveCVAccuracy)c).getParameters());
                else
                    p.writeLine("NoParameterInfo");
                p.writeLine(acc+"");
                for(int j=0;j<data[1].numInstances();j++)
                    p.writeLine(predictions[j][0]+","+predictions[j][1]);

    //            of.writeString(foldAcc[i]+",");

            }catch(Exception e)
            {
                    System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                    e.printStackTrace();
                    System.out.println(" TRAIN "+data[0].relationName()+" has "+data[0].numAttributes()+" attributes and "+data[0].numInstances()+" instances");
                    System.out.println(" TEST "+data[1].relationName()+" has "+data[1].numAttributes()+" attributes and "+data[1].numInstances()+" instances");

                    System.exit(0);
            }
        }
        else
            System.out.println(" Fold "+fold+" already complete");
    }    
    
 /** TRAIN: Generate train predictions for ensemble
  * Input same as test file. This is done separately to test files to make this 
  * compatible with legacy code. It basically loads up the tes
  
  **/
    public static void HESCATrainAccuracyExperiment(String[] args) throws Exception{
//first gives the problem file  
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
                String preds=resultsPath+classifier+"/Predictions"+"/"+problem;
        File f=new File(preds);
        if(!f.exists()){    //Cannot proceed
            
        }
        
//        String problem=unfinished[Integer.parseInt(args[1])-1];
        System.out.println("Classifier ="+classifier+" problem ="+problem);
        Instances all=ClassifierTools.loadData(problemPath+problem+"/"+problem);
        System.out.println(problem+","+classifier+","+all.numInstances()+","+prop*all.numInstances());
        
//Create reproducable split
        Classifier c=setClassifier(classifier);
        
        ((HESCA)c).loadCVWeights(preds+"/internalCV_"+fold+".csv");
        
        OutFile of = new OutFile(resultsPath+classifier+"/Predictions/"+problem+"/trainFold"+fold+".csv");
        Instances[] data=InstanceTools.resampleInstances(all, fold, prop);
//Do a ten fold CV to get an estimate of the ensemble train accuracy.
        double[][] predictions=ClassifierTools.crossValidationWithStats(c, data[0], 10);
        of.writeLine(problem+","+classifier+",train");
        if(c instanceof SaveCVAccuracy)
            of.writeLine(((SaveCVAccuracy)c).getParameters());
        else
            of.writeLine("NoParameterInfo");
        of.writeLine(predictions[0][0]+"");
        for(int j=1;j<predictions[0].length;j++)
            of.writeLine(predictions[0][j]+","+predictions[1][j]);

    }

  
 
 /** PART 2: ***************PROCESS TRAIN AND TEST PREDICTIONS *******/    

 /** WHICH ALGORITHM IS MOST ACCURATE?  collateTestAccuracyResults
     input: filename for results stored in
     resultsPath+filename+".csv"
  generates mean accuracies over all problem files and folds for the components
  as a table, columns are classifiers, rows are problems. These are the results used 
  to generate the critical difference diagram in Section XX Figure XX.
     
     notes: 
     1. Average is over the number of folds present. If there are no folds present a
     at all for a problem, it is ignored. 
     2. The format is a little confused, because
     the internal test accuracy per fold is not recorded, so needs to be calculated. 
     The first column in internalTestPreds is the class value.
     3. Basic diagnostics of what is missing is conducted. It does NOT check each fold is present
     **/        
     public static void collateTestAccuracyResults(String resultName, String sourceDirectory, int nosInEnsemble){
        int folds=100;
        OutFile out=new OutFile(resultsPath+resultName+".csv");
        OutFile diagnostics=new OutFile(resultsPath+sourceDirectory+"MissingFolds.csv");
        
        for(String s:uciFileNames){
            if(new File(resultsPath+sourceDirectory+"/Predictions/"+s).exists()){
                double[] means=new double[nosInEnsemble+1];//Also stores the ensemble results
                double tMean=0;
                int counts=0;
                boolean anyMissing=false;
                for(int i=0;i<folds;i++){
    //Open test file, read in accuracy and test actuals
                    String name=resultsPath+sourceDirectory+"/Predictions/"+s+"/testFold"+i+".csv";
                    File f=new File(name);
                    if(f.exists() && f.length()>0){
    //Open test CV file, read in predictions then calculate accuracy
                        counts++;
                        InFile inf=new InFile(name);
                        int lines=inf.countLines()-3;
                        inf=new InFile(name);
                        inf.readLine();
                        inf.readLine();
                        double acc=inf.readDouble();
                        tMean+=acc;
       //Read in actual and overall pred                 
                        int[][] preds=new int[2][lines];
                        for(int j=0;j<lines;j++){
                            preds[0][j]=inf.readInt();
                            preds[1][j]=inf.readInt();
                        }                 
                        String internalName=resultsPath+sourceDirectory+"/Predictions/"+s+"/internalTestPreds_"+i+".csv";
                        inf=new InFile(internalName);
                        int[][] internalPred=new int[nosInEnsemble+1][lines];
                        double[] accs=new double[nosInEnsemble+1];
                        for(int j=0;j<lines;j++){
                            for(int k=0;k<internalPred.length;k++){
                                internalPred[k][j]=inf.readInt();
                                if(internalPred[k][j]==preds[0][j])
                                    accs[k]++;
                                if(k==0)
                                if(internalPred[0][j]!=preds[0][j]){
                                    System.out.println("ERROR ACTUAL MISMATCH CASE "+j+ " in FOLD "+i);
                                    System.exit(0);
                                }
                                    
                            }
                        }                 
                        for(int k=1;k<internalPred.length;k++)
                            means[k]+=accs[k]/lines;
                    }else{
                            System.out.println(s+"  "+name+" MISSING ");
                        if(!anyMissing){
                            diagnostics.writeString(s+","+i);
                            anyMissing=true;
                        }
                        else
                            diagnostics.writeString(","+i);
                    }
                }
                out.writeString(s+",");
                if(counts>0){
                    out.writeString((tMean/counts)+"");                    
                    for(int k=1;k<nosInEnsemble+1;k++){
                        means[k]/=counts;
                        out.writeString(","+means[k]);
                    }
                }
                out.writeString("\n");
                if(anyMissing)
                    diagnostics.writeString("\n");
                
            }else
                diagnostics.writeLine(s+",MISSING ALL");

        }
    }

/** WHAT IS THE DIFFERENCE IN TRAIN AND TEST ACCURACY 
 
 takes a CV weighted ensemble and 
 *      1. Read in train acc for components from internalCV_i.csv
 *      2. Calculates the test acc for components from  internalTestPreds_i.csv
 *      3. Reads the train acc from ensemble internalCV_i.csv
 *      4. Reads in the test acc for ensemble from testFoldi.csv 
 *      5. Reads in the train acc for ensemble from trainFoldi.csv
 *      6. Collates it all together into a single file
 * TO do: Read in 
 * @param results
 * @param classifier
 * @param fileNames 
 */    
//This is a hack to avoid having to know the number of class values in each problem.       
    static int maxNumClasses=100;
    public static void collateEnsembleDifferenceBetweenTrainTest(String classifier){
        System.out.println("SAVING TO "+resultsPath+"/"+classifier+"TrainTestEnsembleCombined.csv");
        OutFile out=new OutFile(resultsPath+"/"+classifier+"TrainTestEnsembleCombined.csv");
        OutFile out2=new OutFile(resultsPath+"/"+classifier+"DifferencesOnly.csv");
//Final ones
        int numClassifiers=8;   
//        String s="acute-nephritis";
        out2.writeLine(",HESCA,NN,NB,C45,SVML,SVMQ,RandF,RotF,BayesNet");
        for(String s: uciFileNames)
        {
            double meanTestEnsemble=0;
            double meanTrainEnsemble=0;
            double meanDiffEnsemble=0;
            double[] meansTestAcc=new double[numClassifiers];
            double[] meansTrainAcc=new double[numClassifiers];
            double[] meansDiff=new double[numClassifiers];
            double count=0;
            double sumSq=0;
            for(int i=0;i<100;i++){
                double[] trainAcc=new double[numClassifiers];
                double[] testAcc=new double[numClassifiers];
                File f1,f2,f3,f4;
                f1=new File(resultsPath+classifier+"/Predictions/"+s+"/internalCV_"+i+".csv");
                f2= new File(resultsPath+classifier+"/Predictions/"+s+"/internalTestPreds_"+i+".csv");
                f3=new File(resultsPath+classifier+"/Predictions/"+s+"/trainFold_"+i+".csv");
                f4=new File(resultsPath+classifier+"/Predictions/"+s+"/testFold"+i+".csv");
                if(f1.exists() && f1.length()> 0 &&
                   f2.exists() && f2.length()> 0 &&
                   f3.exists() && f3.length()>0 &&
                   f4.exists() && f4.length()>0     ){
                    InFile train=new InFile(resultsPath+classifier+"/Predictions/"+s+"/internalCV_"+i+".csv");
//1. Read in train acc for components from internalCV_i.csv THESE ARE ALSO WEIGHTS FOR Ensemble Train
                    if(train.countLines()==2){   //OK
                        train=new InFile(resultsPath+classifier+"/Predictions/"+s+"/internalCV_"+i+".csv");
                        train.readLine();
                        for(int j=0;j<numClassifiers;j++){
                            trainAcc[j]=train.readDouble();
                            meansTrainAcc[j]+=trainAcc[j];
                        }
                        count++;
//2. Calculates the test acc for components from  internalTestPreds_i.csv                        
    //Annoyingly need to work out test acc for each one. First column is actuals.                    
                        InFile test=new InFile(resultsPath+classifier+"/Predictions/"+s+"/internalTestPreds_"+i+".csv");
                        int lines=test.countLines();
                        test=new InFile(resultsPath+classifier+"/Predictions/"+s+"/internalTestPreds_"+i+".csv");
                        for(int j=0;j<lines;j++){
                            int actual=test.readInt();
                            for(int k=0;k<numClassifiers;k++){
                                int pred=test.readInt();
                                if(actual==pred)
                                    testAcc[k]++;
                            }
                        }
                        for(int k=0;k<numClassifiers;k++){
                            testAcc[k]/=lines;
                            meansTestAcc[k]+=testAcc[k];
                            meansDiff[k]+=testAcc[k]-trainAcc[k];
                        }
//      3. Read in the train acc for ensemble Calculates the train acc for ensemble from trainFold_i.csv
                        InFile trainF=new InFile(resultsPath+classifier+"/Predictions/"+s+"/trainFold_"+i+".csv");
                        trainF.readLine();
                        trainF.readLine();
                        double ensTrainAcc=trainF.readDouble();
//      4. Reads in the test acc for ensemble from testFoldi.csv                         
                        InFile testF=new InFile(resultsPath+classifier+"/Predictions/"+s+"/testFold"+i+".csv");
                        testF.readLine();
                        testF.readLine();
                        double ensTestAcc=testF.readDouble();
                        meanTestEnsemble+=ensTestAcc;
                        meanTrainEnsemble+=ensTrainAcc;
                        meanDiffEnsemble+=ensTestAcc-ensTrainAcc;
                    }
                }else
                    System.out.println("Fold "+i+" one of them doesnt exist "+s);
            }
            out2.writeString(s+","+meanDiffEnsemble/count);
            out.writeString(s+","+count+","+meanDiffEnsemble/count);
            for(int i=0;i<numClassifiers;i++){
                out.writeString(","+meansDiff[i]/count);
                out2.writeString(","+meansDiff[i]/count);
            }
            out.writeString(",,"+meanTrainEnsemble/count);
            for(int i=0;i<numClassifiers;i++)
                out.writeString(","+meansTrainAcc[i]/count);
            out.writeString(",,"+meanTestEnsemble/count);
            for(int i=0;i<numClassifiers;i++)
                out.writeString(","+meansTestAcc[i]/count);
            out.writeString("\n");
            out2.writeString("\n");
        }
        
    }
    
/**Test whether each classifiers mean difference between train/test is significantly
 * different to 0
 * 
 **/
    public static void testsOfDifferenceToZero(String sourceFile){
//Read data        
        InFile data=new InFile(resultsPath+sourceFile);
        int nosProblems=data.countLines()-1;
        data=new InFile(resultsPath+sourceFile);
        String[] temp=data.readLine().split(",");
        int nosClassifiers=temp.length-1;
        String[] names=new String[nosClassifiers];
        for(int i=0;i<nosClassifiers;i++)
            names[i]=temp[i+1];
        double[][] diffs=new double[nosClassifiers][nosProblems];
        for(int j=0;j<nosProblems;j++){
            String t=data.readString(); //Problem name
//            System.out.print("Problem ="+t+",");
            for(int i=0;i<nosClassifiers;i++){
                diffs[i][j]=data.readDouble();
                System.out.print(diffs[i][j]+",");
            }
            System.out.print("\n");            
        }
//Do tests        
       OneSampleTests test=new OneSampleTests();
       for(int i=0;i<nosClassifiers;i++){
          String str=test.performTests(diffs[i]);
          System.out.println("TEST Classifier "+names[i]+":: "+str);
       }
        
    } 
    public static void BVEnsemble(){
//Read in and sort all the predictions
//        algorithm=algo;
        int numClassifiers=9;
        OutFile[] bv=new OutFile[numClassifiers];
        for(int j=0;j<9;j++) {
            bv[j]=new OutFile(resultsPath+"BV/"+algorithm+j+"BV.csv");
            bv[j].writeLine("Problem,accuracy,,biasKohavi,varianceKohavi,,biasDomingos,varianceDomingos,,unbiasedVariance,biasedVariance,netVariance");
        }
        String problem="acute-nephritis";
        TreeSet<String> missing=new TreeSet<>();
        missing.add("miniboone");
        missing.add("plant-margin");
        missing.add("plant-shape");
        missing.add("plant-texture");
        missing.add("trains");
        
        for(int i=0;i<uciFileNames.length;i++)
        {
            problem=uciFileNames[i];
            if(!missing.contains(problem))
            {            
                System.out.println("PROBLEM FILE :"+problem+" Formatting for BV analysis");
//            formatPredictions(problem,100);
                BiasVarianceEvaluation.resultsPath=resultsPath;
                for(int j=0;j<9;j++){ 
                   BVResult b=BiasVarianceEvaluation.findBV(algorithm,problem+j);
                   bv[j].writeLine(problem+","+b.toString());
            }           
            }
        }
        
        
    } 
    
//NOT CONCLUDED  
    public static void formatPredictions(String problem, int folds){
        int numClassifiers=9;//Including the ensemble
        File f=new File(resultsPath+"Folds/"+problem+".csv");
        File f2=new File(resultsPath+algorithm+"/Predictions");
        if(!f.exists()){
            System.out.println("ERROR, file "+f+" not present");
            System.exit(0);
        }
        if(!f2.exists()){
            System.out.println("ERROR, file "+f2+" not present");
            System.exit(0);
        }
//Need to load up testPredictions for each test fold then map them back to the correct original
        
        Instances all=ClassifierTools.loadData(problemPath+problem+"/"+problem);
        int numInstances=all.numInstances();
        int trainSize=(int)(prop*numInstances);
        int testSize=numInstances-trainSize;
        System.out.println(" NUMBER OF INSTANCES="+numInstances);
        System.out.println("Train size ="+trainSize+" test size ="+testSize);
                int [] actualClassVals=new int[numInstances];
        int pos=0;
        for(Instance t:all){
            actualClassVals[pos++]=(int)t.classValue();
//            System.out.println(" Pos ="+(pos-1)+" class val ="+actualClassVals[pos-1]);
        }
//Load mapping            
        InFile allFolds=new InFile(resultsPath+"Folds/"+problem+".csv");
        int[][] resampleMap=new int[folds][numInstances];
        //Load up test fold testPredictions
        ArrayList<Integer>[][] testPredictions;//
        testPredictions = new ArrayList[numInstances][numClassifiers];
        for(int k=0;k<numClassifiers;k++){
            for(int j=0;j<numInstances;j++){
                testPredictions[j][k]=new ArrayList<>();
            }
        }
        
        for(int i=0;i<folds;i++){
//Check the fold actually exists
            File fTest=new File(resultsPath+algorithm+"/Predictions/"+problem+"/testFold"+i+".csv");
            File fTest2=new File(resultsPath+algorithm+"/Predictions/"+problem+"/internalTestPreds_"+i+".csv");
            if(!fTest.exists() || fTest.length()==0)
                System.out.println("Fold "+i+" testFold file is not present, skipping");
            else if(!fTest2.exists() || fTest2.length()==0)
                System.out.println("Fold "+i+" internalTestPreds_ file is not present, skipping");
            else{    
                String str2=allFolds.readLine();
    //            System.out.println("Starting "+str2);
                for(int j=0;j<numInstances;j++){
    // Original index, not needed, but check it should equal j
                   int a=allFolds.readInt();
                   if(a!=j){
                       System.out.println("ERROR in reading in the mapping, misalligned somehow. DEBUG");
                       System.exit(0);
                   }
    //Train/Test, not needed
                   int map=allFolds.readInt();
    //               System.out.println(j+","+a+","+map);
                   resampleMap[i][j]=map;
                   allFolds.readString();//Either TRAIN or TEST in the original split, not needed
                   allFolds.readString();//Either TRAIN or TEST in the new split, not needed
                }

                InFile testPredF=new InFile(resultsPath+algorithm+"/Predictions/"+problem+"/testFold"+i+".csv");
                InFile internalTestPredF=new InFile(resultsPath+algorithm+"/Predictions/"+problem+"/internalTestPreds_"+i+".csv");
                for(int k=0;k<3;k++){ //Remove unrequired header info 
                    testPredF.readLine();
//                    internalTestPredF.readLine();
                }
    //           for(int j=0;j<numInstances;j++)
    //               System.out.println("MAP: "+j+"  -> "+resampleMap[i][j]);

    //Read in the actual and the predicted test into a single array 
                int[][] temp=new int[numInstances][numClassifiers+1];//+1 for actual in position 0
                for(int j=trainSize;j<numInstances;j++){
                    temp[j][0]=testPredF.readInt();//Actual
                    temp[j][1]=testPredF.readInt();//Ensemble predicted
    //                String s=testPredF.readLine(); //Ensemble probs, dont need
                    //Component predictions
                    int act2=internalTestPredF.readInt();
                    if(act2!=temp[j][0]){
                  System.out.println(problem+"FOLD:"+i+" INSTANCE "+j+" CLASS MISSMATCH BETWEEN ENSEMBLE AND COMPONENTS Resample pos ="+j+"  Original position ="+resampleMap[i][j]);
                       System.out.println("Ensemble class ="+temp[j][0]+" component ="+act2);
                       Instance x;
                      System.out.println(" Instance in train ");
                      x=all.instance(resampleMap[i][j]);
                       System.out.println("class ="+x.classValue()+" first val="+x.value(0)+" secoond val ="+x.value(0));
                       System.exit(0);                        
                    }
                    for(int k=2;k<temp[j].length;k++){
                        temp[j][k]=internalTestPredF.readInt();
//                        System.out.println("Fold "+i+" Instance "+(j-trainSize)+" reading in component "+k+" value = "+temp[j][k]);
                    }
                }
    //Split these into the arrays according to the mapping            
               for(int j=0;j<numInstances;j++){
    //Element j
                   int foldPos=resampleMap[i][j]; //This is the position in temp of this instance
    //                  System.out.println(" Element "+j+" in position "+foldPos+" on fold "+i);
                   int actual=temp[j][0];
/*                   if(j>=trainSize && actual!=actualClassVals[foldPos]){
                   System.out.println(problem+" CLASS MISSMATCH Resample pos ="+j+"  Original position ="+foldPos);
                       System.out.println("CLASS MISMATCH IN ACTUAL..... NEEDS DEBUGGING");
                       System.out.println("Instance resample pos ="+j+" in test fold +"+i+" position in original ="+foldPos+" Class from array ="+actualClassVals[foldPos]+" class from TestFold class ="+actual+" instance ="+(j));
                       Instance x;
                      System.out.println(" Instance in train ");
                      x=all.instance(foldPos);
                       System.out.println("class ="+x.classValue()+" first val="+x.value(0));
                       System.exit(0);
                   }
  */                if(j>=trainSize){    //In test fold, save predictions
                       for(int k=1;k<numClassifiers+1;k++)
                           testPredictions[foldPos][k-1].add(temp[j][k]);//Ensemble
                   }
                }
            }
        }
        OutFile[] testF=new OutFile[numClassifiers];
        for(int k=0;k<numClassifiers;k++){
            testF[k]=new OutFile(resultsPath+"BV/"+algorithm+"/"+problem+k+"Test"+".csv");
            for(int j=0;j<all.numInstances();j++){
                testF[k].writeString(j+","+actualClassVals[j]+",");
                for(int in: testPredictions[j][k])
                    testF[k].writeString(","+in);
                testF[k].writeString("\n");
            }
        }
    }
    
    
    
    public static void main(String[] args) throws Exception {
       
        
        
         try{
            if(args.length>0){ //Cluster run
                for (int i = 0; i < args.length; i++) {
                    System.out.println("ARGS ="+i+" = "+args[i]);
                }
                resultsPath="/gpfs/home/ajb/UCIResults/";
                problemPath="/gpfs/home/ajb/UCI ARFF/";
                HESCATrainAccuracyExperiment(args);
            }
            else{         //Local threaded run    
                resultsPath="C:/UCI/UCIResults/";
                problemPath="C:/UCI/UCIData/";
                String classifier="WE";
                String problem="acute-nephritis";
                System.out.println("Problem ="+problem+" Classifier = "+classifier);
                String[] arg={classifier,problem,"1"};
                
//Generate test results for single problem and fold                
//                HESCATestAccuracyExperiment(arg);
//Generate train results for the ensemble and fold                
//                HESCATrainAccuracyExperiment(arg);
//Collate ALL the means into a single file, for overall CD diagram (SEE FIGURE 
//                collateTestAccuracyResults(classifier+"Means", classifier,8);
//Work out ALL the differences between train and test
//                collateEnsembleDifferenceBetweenTrainTest(classifier);
//                testsOfDifferenceToZero("WEDifferencesOnly.csv");                
//Perform a BV decomposition: results in separate files for each classifier
                BVEnsemble();
            }
        }catch(Exception e){
            System.out.println("Exception thrown ="+e);
            e.printStackTrace();
            System.exit(0);
        }       
        
/*
        //        describeData();
//        findAllMappings();
      DataSets.resultsPath="C:/UCI/UCIResults/";
      String root=DataSets.resultsPath;
 //       FormatNewCOTE.collateEnsembleDifferenceBetweenTrainTest("C:/UCI/UCIResults/", "WE", uciFileNames);
//        collateDifferenceBetweenTrainTest("C:/UCI/UCIResults/", "WE", uciFileNames);
        
//        collateResults();
        
        BiasVarianceEvaluation.dataPath="C:/UCI/UCIData/";
        BiasVarianceEvaluation.resultsPath="C:/UCI/UCIResults/";
        BiasVarianceEvaluation.fileNames=uciFileNames;
//        dataPath="/gpfs/home/ajb/TSC Problems/";
//        resultsPath="/gpfs/home/ajb/Results/";
        BVEnsemble();
        
//        fullBV("BOSS");
//        fullBV("TSF");
//       fullBV("RIF_PS_ACF");
 //       fullBV("WE");
        
        
        
        System.exit(0);
*/        
        
    }
    

    public static void makeARFFs(){
        String path="C:\\Data\\UCI Problems All Real Valued\\";
        for(String s:uciFileNames){
            File dir=new File("C:\\Data\\UCI ARFF\\"+s);
            if(!dir.isDirectory())
                dir.mkdir();
            
            File f=new File(path+s+"\\"+s+".arff");
            Instances all;
            if(f.exists()){
                 all=ClassifierTools.loadData(path+s+"\\"+s);
                OutFile of=new OutFile("C:\\Data\\UCI ARFF\\"+s+"\\"+s+".arff");
                of.writeString(all.toString());

//                System.out.println("PRoblem "+test.relationName()+" instances ="+test.numInstances()+" attributes ="+test.numAttributes()+" classes ="+test.numClasses());
            }else{
                System.out.println(s+ "\t\t NO COMBINED FILE, TRY TO MAKE ONE");
                File f2=new File(path+s+"\\"+s+"_test.arff");
                File f3=new File(path+s+"\\"+s+"_train.arff");
                if(f2.exists() && f3.exists()){//Combine
                    all=ClassifierTools.loadData(path+s+"\\"+s+"_train");
                    Instances test=ClassifierTools.loadData(path+s+"\\"+s+"_test");
                    for(Instance ins:test)
                        all.add(ins);
                    OutFile of=new OutFile("C:\\Data\\UCI ARFF\\"+s+"\\"+s+".arff");
                    of.writeString(all.toString());
                }else
                    System.out.println("PROBLEM "+s+" NO ARFF FILE");
            }
            
                
        }
        
    }
    public static void listFiles(){
//Get file directory names
        String path="E:\\Data\\UCI Problems All Real Valued\\";
        File f= new File(path);
        String [] names=f.list();
        for(String s:names)
            System.out.println(s);
        System.out.println("NUMBER OF FOLDERS ="+names.length);
        OutFile out=new OutFile(path+"names.txt");
        out.writeString("{");
        for(String s:names)
            out.writeString("\""+s+"\",");
        out.writeLine("};");  
    }
}
