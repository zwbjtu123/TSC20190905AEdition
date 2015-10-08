/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import fileIO.InFile;
import statistics.simulators.ArmaModel;
import statistics.simulators.SimulateAR;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import net.sourceforge.sizeof.SizeOf;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.spectral_distance_functions.KullbackLeiberDistance;
import weka.core.spectral_distance_functions.LikelihoodRatioDistance;
import weka.core.spectral_distance_functions.LogNormalisedDistance;
import weka.filters.*;
import weka.filters.timeseries.*;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformFactory;

/**
 * @author ajb
 The point of this experiment is to answer the question: what is the best way to deal with 
 ACF transform, given its the optimal transform to use. Alternatives are
 Use all ACF terms. (well, need some maxLag because of the reduction in data available given the lag
 1. Truncate ACF terms (unsupervised).
      need to 1. Define a threshold (significance test) for a single series. 
 e.g. t test, where t= r*sqrt((n-2)/(1-r^2)              
  2/srt(N)
              2. Define whether we treat each series independently or not. e.g. remove when there are no signi 
 2. Filter ACF terms (supervised).
 3. Find best region of the ACF (supervised and unsupervised?) 
 4. Use just the PACF in conjunction with the above
 5. Use the PACF truncated for AR terms with normal method
 * 
 * 
 * 
 * Second Attempt: whats the best way to use ACF for time series classification?
 * On
 * 1. TSC problems
 * 2. AR simulated
 * 3. Shapelet simulated
 * 
 */
public class ACFDomainClassification {
    
    static String resultPath="C:\\Users\\ajb\\Dropbox\\Results\\ACFTest";
    static String dataPath="C:\\Data\\";
    
    public static void transformAllDataSets(String resultsPath, boolean saveTransforms){
        DecimalFormat df = new DecimalFormat("###.###");
        OutFile of = new OutFile(resultsPath);
        System.out.println("************** ACF TRANSFORM ON ALL*******************");
        System.out.println("************** Use concatination of ACF and PACF, both with lag n/4*******************");
        ArrayList<String> names=new ArrayList<>();
        String[] fileNames=DataSets.fileNames;
        for(String s:names){
            of.writeString(s+",");
            System.out.print(s+"\t");
        }
        of.writeString("\n");
        System.out.print("\n");
        for(int i=0;i<fileNames.length;i++)
        {
             Instances test=null;
             Instances train=null;
            try{
                test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileNames[i]+"\\"+fileNames[i]+"_TEST");
                train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileNames[i]+"\\"+fileNames[i]+"_TRAIN");			
                OutFile o2=null,o3=null;
                if(saveTransforms){
                    File f = new File(dataPath+"ACF Transformed TSC Problems\\ACF"+fileNames[i]);
                    if(!f.isDirectory())//Test whether directory exists
                        f.mkdir();
                    o2=new OutFile(dataPath+"ACF Transformed TSC Problems\\ACF"+fileNames[i]+"\\ACF"+fileNames[i]+"_TRAIN.arff");
                    o3=new OutFile(dataPath+"ACF Transformed TSC Problems\\ACF"+fileNames[i]+"\\ACF"+fileNames[i]+"_TEST.arff");
                }
                System.gc();
                System.out.println("Transforming "+fileNames[i]);

                int maxLag=(train.numAttributes()-1)/4;
                Instances allTrain=comboAcfPacf(train,maxLag);
                Instances allTest=comboAcfPacf(test,maxLag);
                dataValidate(train,test);

                if(saveTransforms){
//Need to do this instance by instance to save memory
                    Instances header=new Instances(allTrain,0);
                    o2.writeLine(header.toString());
                    o3.writeLine(header.toString());
                    for(int j=0;j<allTrain.numInstances();j++)
                        o2.writeLine(allTrain.instance(j).toString());
                    for(int j=0;j<allTest.numInstances();j++)
                        o3.writeLine(allTest.instance(j).toString());
                }
    //Save results to file
    //Train Classifiers
                System.out.print(" Classifying ....\n ");
                Classifier[] c= setDefaultSingleClassifiers(names);
                of.writeString(fileNames[i]+",");
                for(int j=0;j<c.length;j++){
                    c[j].buildClassifier(allTrain);
                    double a=utilities.ClassifierTools.accuracy(allTest,c[j]);
                    System.out.print(a+"\t");
                    of.writeString(a+",");
                }                                    
                System.out.print("\n");
                of.writeString("\n");
            }catch(Exception e){
                System.out.println(" Error in accuracy ="+e);
                e.printStackTrace();
                System.exit(0);
            } catch(OutOfMemoryError m){
                System.out.println("OUT OF MEMORY ERROR");
                m.printStackTrace();
                Runtime runtime = Runtime.getRuntime();
                long totalMemory = runtime.totalMemory();
                long freeMemory = runtime.freeMemory();
                long maxMemory = runtime.maxMemory();
                long usedMemory = totalMemory - freeMemory;
    //Summarise memory                    
                System.out.println(" Total ="+totalMemory+" used = "+usedMemory);
                System.out.println(" Problem ="+fileNames[i]);
                try{

                    long testSize=SizeOf.iterativeSizeOf(test);
                    long trainSize=SizeOf.iterativeSizeOf(train);

                    System.out.println("Train set size ="+trainSize);
                    System.out.println("Test set size ="+testSize);                            
                    System.out.println(" USED ="+usedMemory/1000000+" Main Data ="+(testSize+trainSize)/1000000);

                    System.exit(0);
                }catch(Exception e){
                     System.out.println(" Error in memory sizeOf ="+e);
                    e.printStackTrace();
                    System.exit(0);

                }
            }
       }     
    }

        
        
        
public static Classifier[] setSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<>();
		Classifier c;
		sc2.add(new NaiveBayes());
		names.add("NB");
		sc2.add(new J48());
		names.add("C45");
		sc2.add(new IBk(1));
		names.add("NN");
//		c=new DTW_kNN(1);
//		((DTW_kNN)c).setMaxR(0.01);
		
//		sc2.add(c);
//		names.add("NNDTW");
		c=new SMO();
		PolyKernel kernel = new PolyKernel();
		kernel.setExponent(1);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVML");
		c=new SMO();
		kernel = new PolyKernel();
		kernel.setExponent(2);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
                names.add("SVMQ");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(100);
		sc2.add(c);
		names.add("RandF100");
		c=new RotationForest();
		sc2.add(c);
		names.add("RotF30");
//		c=new MultilayerPerceptron();
  //      	sc2.add(c);
//		names.add("Perceptron");
		c=new BayesNet();
        	sc2.add(c);
		names.add("BayesianNetwork");
                
                
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
 // TEST 1   his method tests the base functionality of the class ACF
        public static void basicCorrectnessTests(String path){
  /** This method tests the base functionality of the class ACF, somewhat reproducing the test harness experiments but put heree for tidiness and sanity 
 * affirmation
 * Data is loaded from dropbox folder TSC Problems\\TestData\\ACFTest.arff. This file has four AR(1) series of length 100. Two are phi_1=0.5, two are phi_1=-0.5
 * Externally validated results are in spreadsheet C:\\Research\\Data\\TestData\ACFTests.xls  output is in ACTTestOutput.csv
 * */
      		Instances test=ClassifierTools.loadData(path+"ACFTest");
		DecimalFormat df=new DecimalFormat("##.####");
		ACF acf=new ACF();
		acf.setMaxLag(test.numAttributes()/2);
		try{
                    Instances t2=acf.process(test);
                    System.out.println(" Number of attributes in raw data ="+test.numAttributes());
                    System.out.println(" Number of attributes ACF ="+t2.numAttributes());
                    Instance ins=t2.instance(0);
                    for(int i=0;i<ins.numAttributes()&&i<10;i++)
                            System.out.print(" "+df.format(ins.value(i)));
                    System.out.println("\n TEST 1: Basic test that ACF is calculated correctly with the full calculation. Results stored in ACFTest1Output ");
                    OutFile of=new OutFile(path+"ACFTestOutput1.csv");
                    of.writeString(t2.toString());
    //Test 2: Check the faster normalised calculation. 
                    System.out.println("TEST 2: Check whether the shorthand normalised calculation works");
                    NormalizeCase norm=new NormalizeCase();
                    norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
                    Instances normed=norm.process(test);
                    of=new OutFile(path+"ACFNormalised.csv");
                    of.writeString(normed.toString());
                    acf.setNormalized(true);
                    t2=acf.process(normed);
                    ins=t2.instance(0);
                    for(int i=0;i<ins.numAttributes()&&i<10;i++)
                            System.out.print(" "+df.format(ins.value(i)));
                    of=new OutFile(path+"ACFTestOutput2.csv");
                    of.writeString(t2.toString());

    //Test 3: Check the truncation function. 


                }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
		}
                
                
        }

        
 /**  // TEST 2   compare classifiers on AR data on Time, ACF and PS domains
sanityCheck:Method sanityCheck(): Show all classifiers better on ACF space than raw data or PS
 * 
 * 
 */
        public static void sanityCheck(String file){
            ArrayList<String> names = new ArrayList<String>();
            Classifier[] c=setSingleClassifiers(names);
            double[][] mean=new double[3][c.length];
            double[][] sd=new double[3][c.length];
            
            int minParas=1,maxParas=5,seriesLength=200;
            int nosCases=400;
            int[] nosCasesPerClass={200,200};

            
            OutFile of=new OutFile(resultPath+"\\"+file+".csv");
            OutFile of2=new OutFile(resultPath+"\\"+file+"LatexTable.csv");
            int runs=100;
            for(int i=0;i<runs;i++){
//                if(i%10==0)
                    System.out.println(" run nos ="+i);
                Instances all=SimulateAR.generateARDataSet(minParas,maxParas,seriesLength,nosCasesPerClass,true);
                Instances train,test; 
                all.randomize(new Random());
                try{

                //1. Raw data                
                    train=all;
                    test=new Instances(all);
                    for(int j=0;j<nosCases/2;j++){
                        train.delete(0);
                        test.delete(nosCases/2-1);
                    }
                    System.out.print(" Raw ...");
    //2. ACF Full
                    System.out.print(" ACF full ...");
                    ACF acf=new ACF();
                    Instances acfTrain=acf.process(train);
                    Instances acfTest=acf.process(test);
    //3. PS Full        
                    System.out.print(" PS full ...");
                    PowerSpectrum ps=new PowerSpectrum();
                    Instances psTrain=ps.process(train);
                    Instances psTest=ps.process(test);
                   for(int j=0;j<c.length;j++){ 
                        System.out.print(" Classifier ="+names.get(i));

                        c[j].buildClassifier(train);
                        double a=ClassifierTools.accuracy(test, c[j]);
                        mean[0][j]+=a;
                        sd[0][j]+=a*a;
                        c[j].buildClassifier(acfTrain);
                        a=ClassifierTools.accuracy(acfTest, c[j]);
                        mean[1][j]+=a;
                        sd[1][j]+=a*a;
                        c[j].buildClassifier(psTrain);
                        a=ClassifierTools.accuracy(psTest, c[j]);
                        mean[1][j]+=a;
                        sd[1][j]+=a*a;
                   }
                }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
                }
            }
            for(int i=0;i<3;i++)
            {
                for(int j=0;j<c.length;j++){
                    sd[i][j]=(sd[i][j]-mean[i][j]*mean[i][j]/runs)/runs;
                    mean[i][j]/=runs;
                }
            }
            
            of.writeString(",");
           for(int j=0;j<c.length;j++)
                    of.writeString(names.get(j)+",");
            of.writeString("\n");
            String[] trans={"Time","ACF","PS"};
            for(int i=0;i<3;i++){
                of.writeString(trans[i]+",");
                for(int j=0;j<c.length;j++)
                    of.writeString(mean[i][j]+",");
                of.writeString("\n");
            }
           for(int j=0;j<c.length;j++)
                    of2.writeString(names.get(j)+"\t &");
            of2.writeString("\n");
            for(int i=0;i<3;i++){
                of2.writeString(trans[i]+"\t &");
                for(int j=0;j<c.length;j++)
                    of2.writeString(" "+mean[i][j]+" ("+sd[i][j]+")\t &");
                of2.writeString("\n");
            }
            
            
        }
        
/** Test 3: plot noise vs accuracy for a single classifier **/
        public static void parametersVsTransform(String file,Classifier c){
            int minParas=1,maxParas=1,seriesLength=200;
            int nosCases=400;
            int[] nosCasesPerClass={200,200};
            int runs=10;
            OutFile of=new OutFile(resultPath+"\\"+file+".csv");
            OutFile of2=new OutFile(resultPath+"\\"+file+"LatexTable.csv");
            of.writeLine("Noise,Time,ACFfull,ACFtrun,PSfull,PACFfull,PACFtrunc");
            of2.writeLine("Noise & Time & ACF & PS");
            int paraSteps=20;
            double[][] mean=new double[6][paraSteps];
            double[][] sd=new double[6][paraSteps];
           ArmaModel.setGlobalVariance(1);
           SimulateAR.setMinMaxPara(-0.4,0.4);
           try{
            for(int j=0;j<paraSteps;j++,maxParas+=1){
                    System.out.println(" Running with max paras ="+maxParas);
                    for(int i=0;i<runs;i++){
                    //Generate data AND SET NOISE LEVEL
                        Instances all=SimulateAR.generateARDataSet(minParas,maxParas,seriesLength,nosCasesPerClass,true);
                        Instances train,test; 
                        all.randomize(new Random());

        //1. Raw data                
 //                       System.out.print(" Time ...");
                        train=all;
                        test=new Instances(all);
                        for(int k=0;k<nosCases/2;k++){
                            train.delete(0);
                            test.delete(nosCases/2-1);
                        }
                        c.buildClassifier(train);
                        double a=ClassifierTools.accuracy(test, c);
                        mean[0][j]+=a;
                        sd[0][j]+=a*a;
        //2. ACF Full
//                    System.out.print(" ACF full ...");
                        ACF acf=new ACF();
                        acf.setMaxLag(train.numAttributes()-10);
                        acf.setNormalized(false);
                        Instances acfTrain=acf.process(train);
                        Instances acfTest=acf.process(test);
                        c.buildClassifier(acfTrain);
                        a=ClassifierTools.accuracy(acfTest, c);
                        mean[1][j]+=a;
                        sd[1][j]+=a*a;
                       
        //3. ACF Global truncated
 //                   System.out.print(" ACF trun ...");
                        acf=new ACF();
                        acf.setMaxLag(train.numAttributes()-10);
                        Instances acf2Train=acf.process(train);
                        Instances acf2Test=acf.process(test);
                        int largest=acf.truncate(acf2Train);
//             System.out.println(" LARGEST ="+largest);
                        acf.truncate(acf2Test,largest);
                        c.buildClassifier(acf2Train);
                        a=ClassifierTools.accuracy(acf2Test, c);
                        mean[2][j]+=a;
                        sd[2][j]+=a*a;
        //4. PS Full        
 //                   System.out.print(" PS full ...");
                        PowerSpectrum ps=new PowerSpectrum();
                        Instances psTrain=ps.process(train);
                        Instances psTest=ps.process(test);
                        c.buildClassifier(psTrain);
                        a=ClassifierTools.accuracy(psTest, c);
                        mean[3][j]+=a;
                        sd[3][j]+=a*a;
       //5. PS Truncated
//                        c.buildClassifier(psTrain);
//                        a=ClassifierTools.accuracy(psTest, c);
//                        mean[3][j]+=a;
//                        sd[3][j]+=a*a;
                       
       //5. PACF Full
 //                   System.out.print(" PACF full ...");
                       ARMA pacf=new ARMA();
                       pacf.setMaxLag(train.numAttributes()-10);
                       pacf.setUseAIC(false);
                       Instances pacfTrain=pacf.process(train);
                       Instances pacfTest=pacf.process(test);
                       System.out.println(" PACF num attributes="+pacfTrain.numAttributes());
                        c.buildClassifier(pacfTrain);
                        a=ClassifierTools.accuracy(pacfTest, c);
                        mean[4][j]+=a;
                        sd[4][j]+=a*a;
                                               
       //6. PACF Truncated
/*                  System.out.print(" PACF trunc ...");
                       pacf=new ARMA();
                       pacf.setMaxLag(train.numAttributes()-10);
                       pacf.setUseAIC(false);
                       pacf.setFindCutoffs(true);
                       Instances pacf2Train=pacf.process(train);
                       Instances pacf2Test=pacf.process(test);
                       largest=pacf.truncate(pacf2Train);
                        System.out.println(" LARGEST ="+largest);
                     pacf.truncate(pacf2Test,largest);
                        c.buildClassifier(pacf2Train);
                        a=ClassifierTools.accuracy(pacf2Test, c);
                        mean[5][j]+=a;
                        sd[5][j]+=a*a;
  */                   
                    }
//Calculate mean and SD
                    for(int i=0;i<sd.length;i++)
                    {
                        sd[i][j]=(sd[i][j]-mean[i][j]*mean[i][j]/runs)/runs;
                        mean[i][j]/=runs;
                        System.out.println(" \t "+mean[i][j]);

                    }
                  
   //Write results for noise j to file 
                    of.writeString(maxParas+",");
                    of2.writeString(maxParas+" &\t");
                    for(int i=0;i<mean.length;i++){
                        of.writeString(mean[i][j]+",");
                        of2.writeString(" "+mean[i][j]+" ("+sd[i][j]+")\t &");
                    }
                    of.writeString("\n");
                    of2.writeString("\\\\ \n");
               }
           }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
           }
       }



        public static void classifierComparison(String file){
            ArrayList<String> names=new ArrayList<String>();
            Classifier[] c = setSingleClassifiers(names);
            for(int i=0;i<c.length;i++){
                System.out.println(" Running classifier "+names.get(i));
                parametersVsTransform(names.get(i),c[i]);
            }
                
        }

 
/** NEW SERIES 1:
 *  Which is better, using RAW, FULL ACF, FULL PACF, ARMA, ACF+PACF or ACF+PACF+ARMA?
 * fixed factors: 
 *          classifier. 
 *          data generator: nosCases, nos Classes,model parameters, model variance, model parameter range, 
 *          fitted model: max lag.
 * controlled factors: series length and transform type,  
 *  * 
 */        
        public static void transformComparisonTruncationSimulatedProblems(String path, boolean fixed){
//Nos series constant, increase length, maxlag set tot length/4
            int startLength=100, endLength=1000, increment =100;
            int nosCases=200;
            int[] nosCasesPerClass={nosCases/2,nosCases/2};
            int runs=50;
            int minParas=3;
            int maxParas=3;
            int maxLag;
            double globalVar=1;
            double minParaVal=-0.2;
            double maxParaVal=0.2;
             ArrayList<String> names=new ArrayList<>();
            Classifier[] c=setSingleClassifiers(names);           
           ArmaModel.setGlobalVariance(globalVar);
           SimulateAR.setMinMaxPara(minParaVal,maxParaVal);
           OutFile[] of,of2;
           of=new OutFile[names.size()];
           of2=new OutFile[names.size()];
            Random rand = new Random();
           rand.setSeed(1);

           for(int i=0;i<of.length;i++){
                if(fixed){
                        of[i]=new OutFile(path+names.get(i)+"MeanAccuracyTruncate30.csv");
                        of2[i]=new OutFile(path+names.get(i)+"SDAccuracyTruncate30.csv");
                }
                else{
                        of[i]=new OutFile(path+names.get(i)+"MeanAccuracyTruncateQuarter.csv");
                        of2[i]=new OutFile(path+names.get(i)+"SDAccuracyTruncateQuarter.csv");
                }
           }
           ACF acf = new ACF();
           PACF pacf= new PACF();
           ARMA arma= new ARMA();
           PowerSpectrum ps=new PowerSpectrum();
//           ps.padSeries(true);
           arma.setUseAIC(false);
           int folds=10;
           if(fixed){           
                for(int i=0;i<of.length;i++){
                        of[i].writeLine("\n nosCases,"+nosCases+"model parameters range =["+minParas+","+maxParas+"]"+","
                                + " model variance="+globalVar+" model parameter range =["+minParaVal+","+maxParaVal+"]"+"max lag =30");
                        of2[i].writeLine("\n nosCases,"+nosCases+"model parameters range =["+minParas+","+maxParas+"]"+","
                                + " model variance="+globalVar+" model parameter range =["+minParaVal+","+maxParaVal+"]"+"max lag =30");
                }
           }
           else{
                for(int i=0;i<of.length;i++){
                    of[i].writeLine("\n nosCases,"+nosCases+"model parameters range =["+minParas+","+maxParas+"]"+","
                        + " model variance="+globalVar+" model parameter range =["+minParaVal+","+maxParaVal+"]"+"max lag =length/4");
                    of2[i].writeLine("\n nosCases,"+nosCases+"model parameters range =["+minParas+","+maxParas+"]"+","
                        + " model variance="+globalVar+" model parameter range =["+minParaVal+","+maxParaVal+"]"+"max lag =length/4");
                 }
           }
           for(int i=0;i<of.length;i++){
                of[i].writeLine("Length,PS,ACF,PACF,ARMA,ACF_PACF,ACF_PACF_ARMA");
                of2[i].writeLine("Length,PS,ACF,PACF,ARMA,ACF_PACF,ACF_PACF_ARMA");
           }
            try{
                Instances[] all=new Instances[6];
                for(int length=startLength;length<=endLength;length+=increment){
                    double[][] sum=new double[all.length][c.length];
                    double[][] sumSq=new double[all.length][c.length];
                    if(fixed)
                        maxLag=30;
                    else
                        maxLag=length/4;
                    acf.setMaxLag(maxLag);
                    pacf.setMaxLag(maxLag);
                    arma.setMaxLag(maxLag);
                    System.out.println("\n Running series length ="+length);
                    for(int i=0;i<of.length;i++){
                        of[i].writeString("\n"+length+",");
                        of2[i].writeString("\n"+length+",");
                    }
                    for(int i=0;i<runs;i++){
                    //Generate data AND SET NOISE LEVEL
                        all = new Instances[6];
                        Instances raw=SimulateAR.generateARDataSet(minParas,maxParas,length,nosCasesPerClass,true);
                        raw.randomize(rand);
                        all[0]=ps.process(raw);
//                       ps.truncate(all[0], maxLag);
                        all[1]=acf.process(raw);
                        all[2]=pacf.process(raw);
                        all[3]=arma.process(raw);
                        all[4]=new Instances(all[1]);
                        all[4].setClassIndex(-1);
                        all[4].deleteAttributeAt(all[4].numAttributes()-1); 
//This is a necessary duplication because there seems to be a bug in mergeInstances. If I merge using
// all[2], the classifiers NaiveBayes and C4.5 (and maybe others) crash when trying to sort the instances by attribute.                         
                        Instances temp=pacf.process(raw);
                        all[4]=Instances.mergeInstances(all[4], temp);
                        all[4].setClassIndex(all[4].numAttributes()-1);
                        all[5]=new Instances(all[4]);
                        all[5].setClassIndex(-1);
                        all[5].deleteAttributeAt(all[5].numAttributes()-1);                        
                        temp=arma.process(raw);
                        all[5]=Instances.mergeInstances(all[5], temp);
                        all[5].setClassIndex(all[5].numAttributes()-1);
                   
//                        System.out.println(" Run "+(i+1)+" data generated");
                        for(int j=0;j<all.length;j++){
                            for(int k=0;k<c.length;k++){
                                Evaluation e=new Evaluation(all[j]);
                                e.crossValidateModel(c[k], all[j],folds, rand);                            
                                if(j==1 && k==1)
                                    System.out.print(" "+e.correct()/(double)all[j].numInstances());
                                sum[j][k]+=e.correct()/(double)all[j].numInstances();
                                sumSq[j][k]+=(e.correct()/(double)all[j].numInstances())*(e.correct()/(double)all[j].numInstances());    
                            }
                        }
                    }
                    DecimalFormat df= new DecimalFormat("###.###");
                     System.out.print("\n m="+length);
                    for(int i=0;i<all.length;i++){
                        for(int j=0;j<c.length;j++){
                            sum[i][j]/=runs;
                            sumSq[i][j]=sumSq[i][j]/runs-sum[i][j]*sum[i][j];
                            System.out.print(","+df.format(sum[i][j])+" ("+df.format(sumSq[i][j])+")");
                            of[j].writeString(df.format(sum[i][j])+",");                    
                            of2[j].writeString(df.format(sumSq[i][j])+",");
                        }
                    }
            }
           }catch(Exception e){
               System.out.println(" Error ="+e);
               e.printStackTrace();
               System.exit(0);
           }
                  
            
        }
        
/** Nre Series 2:
 * 
 * 
 */

    /**
     * Nre Series 2:
     */
    public static void transformComparisonTruncationTestProblems(String path, boolean fixed){
//Nos series constant, increase length, maxlag set tot length/4
            int maxLag;
           OutFile of;
           if(fixed)
              of=new OutFile(path+"UCRProblemsTruncate30.csv");
           else
              of=new OutFile(path+"UCRProblemsTruncateQuarterLength.csv");
            Random rand = new Random();
           rand.setSeed(1);

           ACF acf = new ACF();
           PACF pacf= new PACF();
           ARMA arma= new ARMA();
           PowerSpectrum ps=new PowerSpectrum();
//           ps.padSeries(true);
           arma.setUseAIC(true);

           of.writeLine("DataSet,PS,ACF,PACF,ARMA,ACF_PACF,ACF_PACF_ARMA");
           String[] files=DataSets.ucrNames;
            try{
                for(int i=0;i<files.length;i++){
                    System.gc();
                    System.out.println(" Problem file ="+files[i]);
                    Instances[] allTrain = new Instances[6];
                    Instances[] allTest = new Instances[6];
                    Instances temp;
                    Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+files[i]+"\\"+files[i]+"_TRAIN");
                    Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+files[i]+"\\"+files[i]+"_TEST");
                    if(fixed)
                        maxLag=100;
                    else
                       maxLag=(train.numAttributes()-1)/4;
                    train.randomize(rand);
                    acf.setMaxLag(maxLag);
                    pacf.setMaxLag(maxLag);
                    arma.setMaxLag(maxLag);
                    allTrain[0]=ps.process(train);
                    allTest[0]=ps.process(test);
                    ps.truncate(allTrain[0], maxLag);
                    ps.truncate(allTest[0], maxLag);
                    allTrain[1]=acf.process(train);
                    allTest[1]=acf.process(test);
                    allTrain[2]=pacf.process(train);
                    allTest[2]=pacf.process(test);
                    allTrain[3]=arma.process(train);
                    allTest[3]=arma.process(test);

                    allTrain[4]=new Instances(allTrain[1]);
                    allTrain[4].setClassIndex(-1);
                    allTrain[4].deleteAttributeAt(allTrain[4].numAttributes()-1); 
                    temp=pacf.process(train);
                    allTrain[4]=Instances.mergeInstances(allTrain[4], temp);
                    allTrain[4].setClassIndex(allTrain[4].numAttributes()-1);
                    allTest[4]=new Instances(allTest[1]);
                    allTest[4].setClassIndex(-1);
                    allTest[4].deleteAttributeAt(allTest[4].numAttributes()-1); 
                    temp=pacf.process(test);
                    allTest[4]=Instances.mergeInstances(allTest[4], temp);
                    allTest[4].setClassIndex(allTest[4].numAttributes()-1);

                                
                    allTrain[5]=new Instances(allTrain[4]);
                    allTrain[5].setClassIndex(-1);
                    allTrain[5].deleteAttributeAt(allTrain[5].numAttributes()-1);                        
                    temp=arma.process(train);
                    allTrain[5]=Instances.mergeInstances(allTrain[5], temp);
                    allTrain[5].setClassIndex(allTrain[5].numAttributes()-1);
                    allTest[5]=new Instances(allTest[4]);
                    allTest[5].setClassIndex(-1);
                    allTest[5].deleteAttributeAt(allTest[5].numAttributes()-1);                        
                    temp=arma.process(test);
                    allTest[5]=Instances.mergeInstances(allTest[5], temp);
                    allTest[5].setClassIndex(allTest[5].numAttributes()-1);

                    DecimalFormat df= new DecimalFormat("###.###");
                    of.writeString("\n "+files[i]+",");
                    for(int j=0;j<allTrain.length;j++){
                        RandomForest rf=new RandomForest();
                        rf.setNumTrees(200);
                        rf.buildClassifier(allTrain[j]);
                        double a=ClassifierTools.accuracy(allTest[j], rf);
                        of.writeString(df.format(a)+",");
                    }
                }
            }catch(Exception e){
               System.out.println(" Error ="+e);
               e.printStackTrace();
               System.exit(0);
           }
                  
            
        }
        
        
          
        /** REPORTED EXPERIMENT 1:
         * This measures the accuracy of Time, ACF, PS and PACF of a range of classifiers 
         * on simulated AR data. Results for various parameter values are in 
         * 
         * Experiment 1 C:\Users\ajb\Dropbox\Results\ACFTest\Experiment 1 ACF Results.xls
         * 
         * everything tested with 50% truncation. 
         * informal conclusion is that for AR model data
         *  1. For shorter (256) series, PACF beats ACF beats PS (just)
         *  2. Longer series, 512+, PS as good, or better than, others.
         * 3. For longer models, PS better.
         * @param args 
         */
        public static void experiment1FullTransformTests(String file){
            ArrayList<String> names = new ArrayList<String>();
            Classifier[] c=ClassifierTools.setSingleClassifiers(names);
            double[][] mean=new double[4][c.length];
            double[][] sd=new double[4][c.length];
            
            int minParas=1,maxParas=5,seriesLength=256;
            int nosCases=200;
            int[] nosCasesPerClass={nosCases/2,nosCases/2};

            
            OutFile of=new OutFile(resultPath+"\\"+file+".csv");
            OutFile of2=new OutFile(resultPath+"\\"+file+"LatexTable.csv");
            int runs=100;
            for(int i=0;i<runs;i++){
//                if(i%10==0)
                    System.out.println(" run nos ="+i);
                Instances all=SimulateAR.generateARDataSet(minParas,maxParas,seriesLength,nosCasesPerClass,true);
                Instances train,test; 
                all.randomize(new Random());
                try{

                //1. Raw data                
                    train=all;
                    test=new Instances(all);
                    for(int j=0;j<nosCases/2;j++){
                        train.delete(0);
                        test.delete(nosCases/2-1);
                    }
                    System.out.print(" Raw ...");
    //2. ACF Full
                    System.out.print(" ACF full ...");
                    ACF acf=new ACF();
                    acf.setMaxLag(seriesLength/2);
                    Instances acfTrain=acf.process(train);
                    Instances acfTest=acf.process(test);
    //3. PS Full        
                    System.out.print(" PS full ...");
                    PowerSpectrum ps=new PowerSpectrum();
                    Instances psTrain=ps.process(train);
                    Instances psTest=ps.process(test);

                    PACF pacf=new PACF();
                    pacf.setMaxLag(seriesLength/2);
                    Instances pacfTrain=pacf.process(train);
                    Instances pacfTest=pacf.process(test);
                      
                    
                   for(int j=0;j<c.length;j++){ 
                        System.out.print(" Classifier ="+names.get(j));

                        c[j].buildClassifier(train);
                        double a=ClassifierTools.accuracy(test, c[j]);
                        mean[0][j]+=a;
                        sd[0][j]+=a*a;
                        c[j].buildClassifier(acfTrain);
                        a=ClassifierTools.accuracy(acfTest, c[j]);
                        mean[1][j]+=a;
                        sd[1][j]+=a*a;
                        c[j].buildClassifier(psTrain);
                        a=ClassifierTools.accuracy(psTest, c[j]);
                        mean[2][j]+=a;
                        sd[2][j]+=a*a;
                        c[j].buildClassifier(pacfTrain);
                        a=ClassifierTools.accuracy(pacfTest, c[j]);
                        mean[3][j]+=a;
                        sd[3][j]+=a*a;
                   }
                }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
                }
            }
            for(int i=0;i<4;i++)
            {
                for(int j=0;j<c.length;j++){
                    sd[i][j]=(sd[i][j]-mean[i][j]*mean[i][j]/runs)/runs;
                    mean[i][j]/=runs;
                }
            }
            
            of.writeString(",");
           for(int j=0;j<c.length;j++)
                    of.writeString(names.get(j)+",");
            of.writeString("\n");
            String[] trans={"Time","ACF","PS","PACF"};
            for(int i=0;i<trans.length;i++){
                of.writeString(trans[i]+",");
                for(int j=0;j<c.length;j++)
                    of.writeString(mean[i][j]+",");
                of.writeString("\n");
            }
           for(int j=0;j<c.length;j++)
                    of2.writeString(names.get(j)+"\t &");
            of2.writeString("\n");
            for(int i=0;i<trans.length;i++){
                of2.writeString(trans[i]+"\t &");
                for(int j=0;j<c.length;j++)
                    of2.writeString(" "+mean[i][j]+" ("+sd[i][j]+")\t &");
                of2.writeString("\n");
            }
            
            
        }
         /** REPORTED EXPERIMENT 2:
          * 
          * Truncated on standard problems
         */
        public static void experiment2ACFTruncations(String file){
            ArrayList<String> names = new ArrayList<String>();
            Classifier[] c=ClassifierTools.setSingleClassifiers(names);
            double[][] mean=new double[4][c.length];
            double[][] sd=new double[4][c.length];
            
            int minParas=1,maxParas=5,seriesLength=256;
            int nosCases=200;
            int[] nosCasesPerClass={nosCases/2,nosCases/2};

            
            OutFile of=new OutFile(resultPath+"\\"+file+".csv");
            OutFile of2=new OutFile(resultPath+"\\"+file+"LatexTable.csv");
            int runs=10;
            for(int i=0;i<runs;i++){
//                if(i%10==0)
                    System.out.println(" run nos ="+i);
                Instances all=SimulateAR.generateARDataSet(minParas,maxParas,seriesLength,nosCasesPerClass,true);
                Instances train,test; 
                all.randomize(new Random());
                try{

                //1. Raw data: not used for classification for this method               
                    train=all;
                    test=new Instances(all);
                    for(int j=0;j<nosCases/2;j++){
                        train.delete(0);
                        test.delete(nosCases/2-1);
                    }
                    System.out.print(" Raw ...");
    //2. ACF Full
                    System.out.print(" ACF full ...");
                    ACF acf=new ACF();
                    acf.setMaxLag(seriesLength/2);
                    Instances acfTrain=acf.process(train);
                    Instances acfTest=acf.process(test);
    //3. ACF Truncated globally       
                    System.out.print(" ACF global truncation ...");
                    ACF acf2=new ACF();
                    Instances acf2Train=acf2.process(train);
                    Instances acf2Test=acf2.process(test);
                    int d=acf2.truncate(acf2Train,true);
                    System.out.println(" Truncated to "+d+" attributes");
                    acf2.truncate(acf2Test,d);
    //3. ACF Truncated individually       
                    System.out.print(" ACF local truncation after global ...");
                    Instances acf3Train=new Instances(acf2Train);
                    Instances acf3Test=new Instances(acf2Test);
                    d=acf2.truncate(acf3Train,false);
                    acf2.truncate(acf3Test,false);
                      
                  
                   for(int j=0;j<c.length;j++){ 
                        System.out.print(" Classifier ="+names.get(j));
                        int count=0;

                        c[j].buildClassifier(acfTrain);
                        double a=ClassifierTools.accuracy(acfTest, c[j]);
                        mean[count][j]+=a;
                        sd[count++][j]+=a*a;
                        c[j].buildClassifier(acf2Train);
                        a=ClassifierTools.accuracy(acf2Test, c[j]);
                        mean[count][j]+=a;
                        sd[2][j]+=a*a;
                        c[count++].buildClassifier(acf3Train);
                        a=ClassifierTools.accuracy(acf3Test, c[j]);
                        mean[count][j]+=a;
                        sd[count][j]+=a*a;
                 }
                }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
                }
            }
            for(int i=0;i<4;i++)
            {
                for(int j=0;j<c.length;j++){
                    sd[i][j]=(sd[i][j]-mean[i][j]*mean[i][j]/runs)/runs;
                    mean[i][j]/=runs;
                }
            }
            
            of.writeString(",");
           for(int j=0;j<c.length;j++)
                    of.writeString(names.get(j)+",");
            of.writeString("\n");
            String[] trans={"ACF_Full","ACF_TruncateGlobal","ACF_TruncateIndividual"};
            for(int i=0;i<trans.length;i++){
                of.writeString(trans[i]+",");
                for(int j=0;j<c.length;j++)
                    of.writeString(mean[i][j]+",");
                of.writeString("\n");
            }
           for(int j=0;j<c.length;j++)
                    of2.writeString(names.get(j)+"\t &");
            of2.writeString("\n");
            for(int i=0;i<trans.length;i++){
                of2.writeString(trans[i]+"\t &");
                for(int j=0;j<c.length;j++)
                    of2.writeString(" "+mean[i][j]+" ("+sd[i][j]+")\t &");
                of2.writeString("\n");
            }
       
            
        }
        
        public static void experiment2PACFTruncations(String file){
            ArrayList<String> names = new ArrayList<String>();
            Classifier[] c=ClassifierTools.setSingleClassifiers(names);
            double[][] mean=new double[4][c.length];
            double[][] sd=new double[4][c.length];
            
            int minParas=1,maxParas=5,seriesLength=256;
            int nosCases=200;
            int[] nosCasesPerClass={nosCases/2,nosCases/2};

            
            OutFile of=new OutFile(resultPath+"\\"+file+".csv");
            OutFile of2=new OutFile(resultPath+"\\"+file+"LatexTable.csv");
            int runs=10;
            for(int i=0;i<runs;i++){
//                if(i%10==0)
                    System.out.println(" run nos ="+i);
                Instances all=SimulateAR.generateARDataSet(minParas,maxParas,seriesLength,nosCasesPerClass,true);
                Instances train,test; 
                all.randomize(new Random());
                try{

                //1. Raw data: not used for classification for this method               
                    train=all;
                    test=new Instances(all);
                    for(int j=0;j<nosCases/2;j++){
                        train.delete(0);
                        test.delete(nosCases/2-1);
                    }
                    System.out.print(" Raw ...");
    //2. ACF Full
                    System.out.print(" ACF full ...");
                    ACF acf=new ACF();
                    acf.setMaxLag(seriesLength/2);
                    Instances acfTrain=acf.process(train);
                    Instances acfTest=acf.process(test);
    //3. ACF Truncated globally       
                    System.out.print(" ACF global truncation ...");
                    ACF acf2=new ACF();
                    Instances acf2Train=acf2.process(train);
                    Instances acf2Test=acf2.process(test);
                    int d=acf2.truncate(acf2Train,true);
                    System.out.println(" Truncated to "+d+" attributes");
                    acf2.truncate(acf2Test,d);
    //3. ACF Truncated individually       
                    System.out.print(" ACF local truncation after global ...");
                    Instances acf3Train=new Instances(acf2Train);
                    Instances acf3Test=new Instances(acf2Test);
                    d=acf2.truncate(acf3Train,false);
                    acf2.truncate(acf3Test,false);
                      
                  
                   for(int j=0;j<c.length;j++){ 
                        System.out.print(" Classifier ="+names.get(j));
                        int count=0;

                        c[j].buildClassifier(acfTrain);
                        double a=ClassifierTools.accuracy(acfTest, c[j]);
                        mean[count][j]+=a;
                        sd[count++][j]+=a*a;
                        c[j].buildClassifier(acf2Train);
                        a=ClassifierTools.accuracy(acf2Test, c[j]);
                        mean[count][j]+=a;
                        sd[2][j]+=a*a;
                        c[count++].buildClassifier(acf3Train);
                        a=ClassifierTools.accuracy(acf3Test, c[j]);
                        mean[count][j]+=a;
                        sd[count][j]+=a*a;
                 }
                }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
                }
            }
            for(int i=0;i<4;i++)
            {
                for(int j=0;j<c.length;j++){
                    sd[i][j]=(sd[i][j]-mean[i][j]*mean[i][j]/runs)/runs;
                    mean[i][j]/=runs;
                }
            }
            
            of.writeString(",");
           for(int j=0;j<c.length;j++)
                    of.writeString(names.get(j)+",");
            of.writeString("\n");
            String[] trans={"ACF_Full","ACF_TruncateGlobal","ACF_TruncateIndividual"};
            for(int i=0;i<trans.length;i++){
                of.writeString(trans[i]+",");
                for(int j=0;j<c.length;j++)
                    of.writeString(mean[i][j]+",");
                of.writeString("\n");
            }
           for(int j=0;j<c.length;j++)
                    of2.writeString(names.get(j)+"\t &");
            of2.writeString("\n");
            for(int i=0;i<trans.length;i++){
                of2.writeString(trans[i]+"\t &");
                for(int j=0;j<c.length;j++)
                    of2.writeString(" "+mean[i][j]+" ("+sd[i][j]+")\t &");
                of2.writeString("\n");
            }
       
            
        }
        

        public static void threeDomainsExample(String file){
            int length=128;
            double[] time=new double[length];
            for(int i=0;i<length;i++)
                time[i]=Math.sin(Math.PI*((double)i/50.0))/10+Math.random();
            
            
            double[] acf=ACF.fitAutoCorrelations(time,100);
            double[] ps=PowerSpectrum.powerSpectrum(time);
            OutFile of=new OutFile(file);
            for(int i=0;i<100;i++)
                of.writeLine(time[i]+","+acf[i]+","+ps[i]);
        }
        
        public static class Pair{
            public double mean;
            public double sd;
        }

        public static Instances comboAcfPacf(Instances data, int length){
            ACF acf = new ACF();
            PACF pacf= new PACF();
            ARMA ar=new ARMA();
            ar.setUseAIC(true);
            int maxLag=length;
            acf.setMaxLag(maxLag);
            pacf.setMaxLag(maxLag);
            ar.setMaxLag(maxLag);
            Instances combo=null;
            acf.setNormalized(false);
            try{
                Instances acfData=acf.process(data);
                Instances pacfData=pacf.process(data);
                combo=new Instances(acfData);
                combo.setClassIndex(-1);
                combo.deleteAttributeAt(combo.numAttributes()-1); 
                combo=Instances.mergeInstances(combo, pacfData);
                combo.deleteAttributeAt(combo.numAttributes()-1); 
               Instances arData=ar.process(data);                
               combo=Instances.mergeInstances(combo, arData);
               combo.setClassIndex(combo.numAttributes()-1);
               
            }catch(Exception e){
                System.out.println(" ERROR in ACF Transform ="+e);
                System.exit(0);
            }
            return combo;
        }
        public static void dataValidate(Instances train, Instances test){
//Check for NaNs, missing values and infinity in both
            int i=0;
            System.out.println(" CHECKING DATA SET "+train.relationName());
            
            while(i<train.numAttributes()-1){
                if(singleValueAttribute(train,i)){
                    System.out.println(" Deleting Attribute "+train.attribute(i).name());
                    train.deleteAttributeAt(i);
                    test.deleteAttributeAt(i);
                }
                else
                    i++;
            }
            
            
        }
        
    public static void transformOnCluster(String fileName){
        String directoryName="TSC_Problems";
        String resultName="ChangeTransformedTSC_Problems";
        DecimalFormat df = new DecimalFormat("###.###");
        System.out.println("************** ACF TRANSFORM ON "+fileName+"   *******************");
        Instances test=null;
        Instances train=null;
        test=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/"+fileName+"/"+fileName+"_TEST");
        train=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/"+fileName+"/"+fileName+"_TRAIN");			
        OutFile o2=null,o3=null;
        o2=new OutFile(DataSets.clusterPath+resultName+"/Change"+fileName+"/Change"+fileName+"_TRAIN.arff");
        o3=new OutFile(DataSets.clusterPath+resultName+"/Change"+fileName+"/Change"+fileName+"_TEST.arff");
        Instances header;
        try{
            int maxLag=(train.numAttributes()-1)/4;
            if(maxLag>100)
                maxLag=100;
            Instances allTrain=comboAcfPacf(train,maxLag);
            Instances allTest=comboAcfPacf(test,maxLag);
            dataValidate(train,test);
            header=new Instances(allTrain,0);
            o2.writeLine(header.toString());
            for(int j=0;j<allTrain.numInstances();j++)
                o2.writeLine(allTrain.instance(j).toString());
            o3.writeLine(header.toString());
            for(int j=0;j<allTest.numInstances();j++)
                o2.writeLine(allTest.instance(j).toString());
            
        }catch(Exception e){
            System.err.println("Exception "+e+" creating train transform for "+fileName);
            e.printStackTrace();
            System.exit(0);
        }
    }
     public static void transformOnDropbox(String fileName){
        String directoryName="TSC Problems";
        String resultName="Change Transformed TSC Problems";
        DecimalFormat df = new DecimalFormat("###.###");
        System.out.println("************** ACF TRANSFORM ON "+fileName+"   *******************");
        Instances test=null;
        Instances train=null;
        test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+directoryName+"/"+fileName+"/"+fileName+"_TEST");
        train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+directoryName+"/"+fileName+"/"+fileName+"_TRAIN");			
        OutFile o2=null,o3=null;
        
        File f = new File(DataSets.dropboxPath+resultName+"/Change"+fileName);
        if(!f.isDirectory())//Test whether directory exists
             f.mkdir();        
        o2=new OutFile(DataSets.dropboxPath+resultName+"/Change"+fileName+"/Change"+fileName+"_TRAIN.arff");
        o3=new OutFile(DataSets.dropboxPath+resultName+"/Change"+fileName+"/Change"+fileName+"_TEST.arff");

//Change transform: ACF, PACF and AR 1/6 original length
// For normal data, first difference and second difference.        
        
        
        Instances header;
        try{
            int seriesLength=train.numAttributes()-1;
            ACF acf=new ACF();
            PACF pacf=new PACF();
            ARMA arma= new ARMA();
            acf.setMaxLag(seriesLength/3);
            pacf.setMaxLag(seriesLength/3);
            arma.setUseAIC(false);
            arma.setMaxLag(seriesLength/3);
            
    //Size transformed train data
            
            Instances acfTrain=acf.process(train);
            Instances acfTest=acf.process(test);
            Instances pacfTrain=pacf.process(train);
            Instances pacfTest=pacf.process(test);
            Instances armaTrain=arma.process(train);
            Instances armaTest=arma.process(test);
       
            
            acfTrain.setClassIndex(-1);
            acfTrain.deleteAttributeAt(acfTrain.numAttributes()-1);
            pacfTrain.setClassIndex(-1);
            pacfTrain.deleteAttributeAt(pacfTrain.numAttributes()-1);
            
            Instances fullTrain=Instances.mergeInstances(acfTrain, pacfTrain);
            fullTrain=Instances.mergeInstances(fullTrain, armaTrain);
            
            acfTest.setClassIndex(-1);
            acfTest.deleteAttributeAt(acfTest.numAttributes()-1);
            pacfTest.setClassIndex(-1);
            pacfTest.deleteAttributeAt(pacfTest.numAttributes()-1);
            
            Instances fullTest=Instances.mergeInstances(acfTest, pacfTest);
            fullTest=Instances.mergeInstances(fullTest, armaTest);
            
            
            header=new Instances(fullTrain,0);
            o2.writeLine(header.toString());
            for(int j=0;j<fullTrain.numInstances();j++)
                o2.writeLine(fullTrain.instance(j).toString());
            o3.writeLine(header.toString());
            for(int j=0;j<fullTest.numInstances();j++)
                o3.writeLine(fullTest.instance(j).toString());
            
        }catch(Exception e){
            System.err.println("Exception "+e+" creating train transform for "+fileName);
            e.printStackTrace();
            System.exit(0);
        }
    }   

   
     
     public static Classifier[] setDefaultSingleClassifiers(ArrayList<String> names){
            ArrayList<Classifier> sc2=new ArrayList<>();
            kNN k = new kNN(100);
            k.setCrossValidate(true);
            k.normalise(false);
            k.setDistanceFunction(new EuclideanDistance());
            sc2.add(k);
            names.add("kNN_Euclid");
    
            
            Classifier c;
            sc2.add(new NaiveBayes());
            names.add("NB");
            sc2.add(new J48());
            names.add("C45");
            c=new SMO();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            ((SMO)c).setKernel(kernel);
            sc2.add(c);
            names.add("SVML");
            c=new SMO();
/*            kernel = new PolyKernel();
            kernel.setExponent(2);
            ((SMO)c).setKernel(kernel);
            sc2.add(c);
            names.add("SVMQ");
            c=new RandomForest();
            ((RandomForest)c).setNumTrees(200);
            sc2.add(c);
            names.add("RandF200");
            RotationForest f=new RotationForest();
            f.setNumIterations(50);
            sc2.add(f);
            names.add("RotF50");
*/
            Classifier[] sc=new Classifier[sc2.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=sc2.get(i);

            return sc;
    } 

     
    public static void buildClassifier(String fileName){
        String directoryName="Change Transformed TSC Problems";
        Instances test=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/Change"+fileName+"/Change"+fileName+"_TEST");
        Instances train=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/Change"+fileName+"/Change"+fileName+"_TRAIN");			
        ArrayList<String> names= new ArrayList<>();
        Classifier[] c =setDefaultSingleClassifiers(names); 
        WeightedEnsemble    w=new WeightedEnsemble(c,names);
        OutFile of = new OutFile("change/"+fileName+"ChangeAcc.csv");
            try{
                w.buildClassifier(train);
                double a=ClassifierTools.accuracy(test, w);
                System.out.println(fileName+"\t Accuracy ="+a);
                of.writeString(fileName+","+a);
            }catch(Exception e){
               System.err.println("Exception = "+e);
               e.printStackTrace();
               System.exit(0);
            }
        
    }    
    public static void combineResults(String file){
        OutFile of = new OutFile(file);
        for(String s: DataSets.fileNames){
            File fi=new File("C:\\Research\\Papers\\2015\\TKDE COTE Tony\\acfoutput\\ACF_Results"+s+".csv");
            if(fi.exists()){
                InFile f= new InFile("C:\\Research\\Papers\\2015\\TKDE COTE Tony\\acfoutput\\ACF_Results"+s+".csv");
                of.writeLine(f.readLine());
            }
        } 
        for(String s: DataSets.fileNames){
            File fi=new File("C:\\Research\\Papers\\2015\\TKDE COTE Tony\\acfoutput\\ACF_ResultsCombo"+s+".csv");
            if(fi.exists()){
                InFile f= new InFile("C:\\Research\\Papers\\2015\\TKDE COTE Tony\\acfoutput\\ACF_ResultsCombo"+s+".csv");
                of.writeLine(f.readLine());
            }
        } 
    }
    public static void evalTransforms(String fileName, OutFile of, boolean onCluster){ 
        Instances test=null;
        Instances train=null;
        if(!onCluster){
            test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileName+"\\"+fileName+"_TEST");
            train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileName+"\\"+fileName+"_TRAIN");			
        }
        else{
            test=utilities.ClassifierTools.loadData(DataSets.clusterPath+fileName+"/"+fileName+"_TEST");
            train=utilities.ClassifierTools.loadData(DataSets.clusterPath+fileName+"/"+fileName+"_TRAIN");			
        }
            
        int maxLag=(train.numAttributes()-1)/4;
        if(maxLag>100)
            maxLag=100;
 //       if(maxLag<10)
 //           maxLag=10;
        
        Instances arTrain,arTest;
        Instances acfTrain,acfTest;
        Instances pacfTrain,pacfTest;
        Instances comboTrain,comboTest;
        ACF acf = new ACF();
        PACF pacf= new PACF();
        ARMA ar=new ARMA();
        ar.setUseAIC(true);
        acf.setMaxLag(maxLag);
        pacf.setMaxLag(maxLag);
        ar.setMaxLag(maxLag);
        acf.setNormalized(false);
        try{
            acfTrain=acf.process(train);
            arTrain=ar.process(train);
            pacfTrain=pacf.process(train);
   /*        comboTrain=new Instances(acfTrain);
            comboTrain.setClassIndex(-1);
            comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
            comboTrain=Instances.mergeInstances(comboTrain, pacfTrain);
            comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
            comboTrain=Instances.mergeInstances(comboTrain, arTrain);
            comboTrain.setClassIndex(comboTrain.numAttributes()-1);
/* 
            System.out.println("PACF Train num atts= "+pacfTrain.numAttributes()+" num insts = "+pacfTrain.numInstances());
            for(int j=0;j<pacfTrain.numInstances();j++){
                Instance ins=pacfTrain.instance(j);
                System.out.print("\n PACF Train +"+j+" num atts= "+ins.numAttributes()+"  ");
                for(int i=0;i<ins.numAttributes();i++)
                    System.out.print(ins.value(i)+",");
            }
            
            for(int i=0;i<pacfTrain.numAttributes();i++)
                            System.out.println(i+"  Attribute "+pacfTrain.attribute(i)+" has index ="+pacfTrain.attribute(i).index()+" and isnumeric = "+pacfTrain.attribute(i).isNumeric());
            System.out.println("PACF full data set = "+pacfTrain.toString());
*/
            acfTest=acf.process(test);
            arTest=ar.process(test);
            pacfTest=pacf.process(test);
/*            comboTest=new Instances(acfTest);
            comboTest.setClassIndex(-1);
            comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
            comboTest=Instances.mergeInstances(comboTest, pacfTest);
            comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
            comboTest=Instances.mergeInstances(comboTest, arTest);
            comboTest.setClassIndex(comboTest.numAttributes()-1);
  */        WeightedEnsemble we= new WeightedEnsemble();
            we.buildClassifier(acfTrain);
            double a1=ClassifierTools.accuracy(acfTest, we);
           we= new WeightedEnsemble();
            we.buildClassifier(arTrain);
            double a2=ClassifierTools.accuracy(arTest, we);
            we= new WeightedEnsemble();
            we.buildClassifier(pacfTrain);
            double a3=ClassifierTools.accuracy(pacfTest, we);
            we= new WeightedEnsemble();
//            we.buildClassifier(comboTrain);
//            double a4=ClassifierTools.accuracy(comboTest, we);
            System.out.println(fileName+","+a1+","+a2+","+a3);
            of.writeLine(fileName+","+a1+","+a2+","+a3);
        }catch(Exception e){
            System.out.println(" ERROR in ACF Combo experiment ="+e);
            e.printStackTrace();
            System.exit(0);
        }
        Instances fullTrain=comboAcfPacf(train,maxLag);                            
        Instances fullTest=comboAcfPacf(test,maxLag);   
    }
    

    public static void evalComboTransforms(String fileName, OutFile of, boolean onCluster){ 
        Instances test;
        Instances train;
        if(!onCluster){
            test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileName+"\\"+fileName+"_TEST");
            train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileName+"\\"+fileName+"_TRAIN");			
        }
        else{
            test=utilities.ClassifierTools.loadData(DataSets.clusterPath+fileName+"/"+fileName+"_TEST");
            train=utilities.ClassifierTools.loadData(DataSets.clusterPath+fileName+"/"+fileName+"_TRAIN");			
        }
            
        int maxLag=(train.numAttributes()-1)/4;
        if(maxLag>100)
            maxLag=100;
 //       if(maxLag<10)
 //           maxLag=10;
        
        Instances arTrain,arTest;
        Instances acfTrain,acfTest;
        Instances pacfTrain,pacfTest;
        Instances comboTrain,comboTest;
        ACF acf = new ACF();
        PACF pacf= new PACF();
        ARMA ar=new ARMA();
        ar.setUseAIC(true);
        acf.setMaxLag(maxLag);
        pacf.setMaxLag(maxLag);
        ar.setMaxLag(maxLag);
        acf.setNormalized(false);
        try{
            acfTrain=acf.process(train);
            arTrain=ar.process(train);
            pacfTrain=pacf.process(train);
           comboTrain=new Instances(acfTrain);
            comboTrain.setClassIndex(-1);
            comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
            comboTrain=Instances.mergeInstances(comboTrain, pacfTrain);
            comboTrain.deleteAttributeAt(comboTrain.numAttributes()-1); 
            comboTrain=Instances.mergeInstances(comboTrain, arTrain);
            comboTrain.setClassIndex(comboTrain.numAttributes()-1);
 
            acfTest=acf.process(test);
            arTest=ar.process(test);
            pacfTest=pacf.process(test);
            comboTest=new Instances(acfTest);
            comboTest.setClassIndex(-1);
            comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
            comboTest=Instances.mergeInstances(comboTest, pacfTest);
            comboTest.deleteAttributeAt(comboTest.numAttributes()-1); 
            comboTest=Instances.mergeInstances(comboTest, arTest);
            comboTest.setClassIndex(comboTest.numAttributes()-1);
            WeightedEnsemble we= new WeightedEnsemble();
            we.buildClassifier(acfTrain);
            we.buildClassifier(comboTrain);
            double a4=ClassifierTools.accuracy(comboTest, we);
            System.out.println(fileName+","+a4);
            of.writeLine(fileName+","+a4);
        }catch(Exception e){
            System.out.println(" ERROR in ACF Combo experiment ="+e);
            e.printStackTrace();
            System.exit(0);
        }
        Instances fullTrain=comboAcfPacf(train,maxLag);                            
        Instances fullTest=comboAcfPacf(test,maxLag);   
    }
    
    
    
    public static void mergeFiles(){
        String path="C:\\Users\\ajb\\Dropbox\\Results\\ChangeDomain\\EnsembleAccuracy\\";
        String[] files=DataSets.fileNames;
        OutFile result=new OutFile(path+"TestAcc.csv");
        InFile f;
        for(int i=0;i<files.length;i++){
             File file = new File(path+files[i]+"ChangeAcc.csv");
             if(file.exists()){
                f=new InFile(path+files[i]+"ChangeAcc.csv");
                String s=f.readLine();
                result.writeLine(s);
             }
        }
    }

//                 int index=Integer.parseInt(args[0])-1;
     
    public static boolean singleValueAttribute(Instances d, int p){
        for(int i=0;i<d.numInstances()-1;i++){
            if(d.instance(i).value(p)!=d.instance(i+1).value(p))
                return false;
        }
        return true;
        
    }
    
    public static void testACF(){
             System.out.println("************** ACF/PACF TRANSFORM ON ALL*******************");
            System.out.println("************** Use concatination of ACF and PACF, both with lag n/2*******************");
            ArrayList<String> names=new ArrayList<>();
            String fileName="ElectricDevices";
                     Instances test=null;
                     Instances train=null;
                    try{
                           test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileName+"\\"+fileName+"_TEST");
                            train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fileName+"\\"+fileName+"_TRAIN");			
                            OutFile o2=null,o3=null;
                            int maxLag=(train.numAttributes()-1)/4;
                            train=comboAcfPacf(train,maxLag);                            
                            test=comboAcfPacf(test,maxLag);   
//Truncate                             
                            
//Output first ACF function
                            for(int i=0;i<train.instance(0).numAttributes()-1;i++)
                                if(i<(train.instance(0).numAttributes()-1)/2)
                                 System.out.println(" lag ="+(i+1)+" ACF ="+train.instance(0).value(i));    
                            else
                                 System.out.println(" lag ="+(train.instance(0).numAttributes()-1-i)+" PACF ="+train.instance(0).value(i));    
//Save problem files 
                            System.out.print(" Save to file ....\n ");
                            OutFile ofTrain =new OutFile("C:\\Data\\ACF Transformed TSC Problems\\ACFElectricDevices_TRAIN.arff");
                            ofTrain.writeString(train.toString());
                            OutFile ofTest =new OutFile("C:\\Data\\ACF Transformed TSC Problems\\ACFElectricDevices_TEST.arff");
                            ofTest.writeString(test.toString());
                            ofTrain.closeFile();
                            ofTest.closeFile();
    //Train Classifiers
                            
                            System.out.print(" Classifying ....\n ");
                            Classifier[] c= setDefaultSingleClassifiers(names);
                            for(int j=0;j<c.length;j++){
                                c[j].buildClassifier(train);
                                double a=utilities.ClassifierTools.accuracy(test,c[j]);
                                System.out.print(a+"\t");
                            }                                    
                    }catch(Exception e){
                            System.out.println(" Error in accuracy ="+e);
                            e.printStackTrace();
                            System.exit(0);
                    }   }
  
       public static void ACFvsARvsPACF(int seriesLength,OutFile of,OutFile of2){
            int minParas=1,maxParas=4;
            int nosCases=400;
            int[] nosCasesPerClass={200,200};
            int runs=30;
            WeightedEnsemble c;
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
        //2. ACF
//                    System.out.print(" ACF full ...");
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
                       
        //2. ARMA ACF Global truncated
 //                   System.out.print(" ACF trun ...");
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
                    for(int i=0;i<sd.length;i++)
                    {
                        sd[i]=(sd[i]-mean[i]*mean[i]/runs)/runs;
                        mean[i]/=runs;
                        System.out.println(" \t "+mean[i]);

                    }
                  
   //Write results to file
                    of.writeString(seriesLength+",");
                    of2.writeString(seriesLength+" &\t");
                    for(int i=0;i<mean.length;i++){
                        of.writeString(mean[i]+",");
                        of2.writeString(" "+mean[i]+" ("+sd[i]+")\t &");
                    }
                    for(int i=0;i<mean.length;i++){
                        of.writeString(sd[i]+",");
                    }
                    of.writeString("\n");
                    of2.writeString("\\\\ \n");
               
           }catch(Exception e){
			System.out.println(" Exception in ACF harness="+e);
			e.printStackTrace();
                        System.exit(0);
           }
       }


    
    public static void mergeSimResults(String file){
        String path="C:\\Research\\Papers\\2015\\TKDE COTE Tony\\acfsimoutput\\";
        OutFile combo=new OutFile(path+file);
        for(int i=100;i<=900;i+=10){
            InFile f = new InFile(path+"ACFSim"+i+".csv");
            combo.writeLine(f.readLine());
        }
        
    }
     public static void main(String[] args){
          if(args[0]!=null){
            int index=Integer.parseInt(args[0])-1;
            index=100+10*index;
            OutFile of = new OutFile("acfsimoutput/ACFSim"+index+".csv");
            OutFile of2=new OutFile("acfsimoutput/ACFSimLatex"+index+".tex");
    //            of.writeLine(seriesLength+",ACF,AR,PACF,COMBO");
     //           of2.writeLine(seriesLength+" &  ACF & AR & PACF & COMBO\\");
            ACFvsARvsPACF(index,of,of2);
         }
         System.exit(0);
//         System.out.println(" Nos problems = "+DataSets.fileNames.length);
         if(args[0]!=null){
             int index=Integer.parseInt(args[0])-1;
             OutFile of = new OutFile("acfoutput/ACF_ResultsCombo"+DataSets.fileNames[index]+".csv");
            evalComboTransforms(DataSets.fileNames[index],of,true);
            //evalTransforms(DataSets.fileNames[index],of,true);
         }else{
             OutFile of = new OutFile("ACF_Results.csv");
            for(int i=0;i<DataSets.fileNames.length;i++)
                evalTransforms(DataSets.fileNames[i],of,false);
         }
         System.exit(0);
             
         
//          testACF();     
    transformComparisonTruncationTestProblems("C:\\Users\\ajb\\Dropbox\\Results\\ChangeDomain\\ACF_PACFTest.csv",false);         
  //     transformAllDataSets("C:\\Users\\ajb\\Dropbox\\Results\\ChangeDomain\\ACF_PACFTest.csv",true);
transformComparisonTruncationSimulatedProblems("C:\\Users\\ajb\\Dropbox\\Results\\ACFDomain\\",false);
         /*
         testACF();       
         double[] d={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,17,331,18,18,15,289,145,0,0,0,0,0,0,0,0};
                           ACF acf = new ACF();
                            PACF pacf= new PACF();
                            int maxLag=24;
                    acf.setMaxLag(maxLag);
                    acf.setNormalized(false);
                    double[] a=acf.fitAutoCorrelations(d);
                    System.out.println(" ACF 1 = "+a[0]);
                    System.out.println(" ACF 2 = "+a[1]);
         */
//            for(int i=0;i<DataSets.fileNames.length;i++)
//                transformOnDropbox(DataSets.fileNames[i]);
//mergeFiles();
         
 //         int index=Integer.parseInt(args[0])-1;
  //      buildClassifier(DataSets.fileNames[index]);
//            System.out.println("Input ="+index);
  //         transformOnCluster(DataSets.fileNames[index]);
//                transformComparisonTruncationTestProblems("C:\\Users\\ajb\\Dropbox\\Results\\ACFTest\\", true);
//                transformComparisonTruncationTestProblems("C:\\Users\\ajb\\Dropbox\\Results\\ACFTest\\", false);
 
 //           basicCorrectnessTests("C:\\Users\\ajb\\Dropbox\\TSC Problems\\TestData\\");
            //       experiment1FullTransformTests("experiment1FullTransform");
//        experiment2ACFTruncations("experiment2ACFTRuncations");
//            sanityCheck("sanityCheck");
//        parametersVsTransform("parametersVsTransform", new RandomForest());
//            classifierComparison("classifierCompare");
//            threeDomainsExample("C:\\Admin\\Research\\Presentations\\UEA Talks\\Example.csv");
                        
}		


}
