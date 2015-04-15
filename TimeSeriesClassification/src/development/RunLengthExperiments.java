package development;

import weka.core.spectral_distance_functions.LikelihoodRatioDistance;
import java.util.*;
import java.text.*;

import statistics.simulators.SimulateAR;
import weka.core.*;
import weka.filters.*;
import weka.filters.timeseries.*;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.*;
import fileIO.*;
import utilities.ClassifierTools;
import weka.classifiers.trees.*;
import weka.classifiers.bayes.*;
import weka.classifiers.functions.Logistic;
import weka.core.elastic_distance_measures.*;


/*
 * Experimental agenda
 * 1. Show DTW doesnt work on ARMA data
 * 2. Show ARMA doesnt work on TSDM?
 * 3. Show that RLE tends towards ARMA for sim data
 * 4. Show that RLE outperforms FFT?
 * 5. Evaluate alternative distance metrics for run lengths
 * 6. Run a model shift experiment
 */


public class RunLengthExperiments {
 		public static String path="C:\\Research\\Data\\Time Series Classification\\";
   
 /* Problems where ACF oputperforms 1-NN */   
    		public static String[] fileNames={	//Number of train,test cases,length,classes
				"OSULeaf", //200,242,427,6
				"SwedishLeaf", //500,625,128,15
				"wafer",//1000,6174,152,2
				//Index 18, after this the data has not been normalised.
				"Beef", //30,30,470,5
				"Coffee", //28,28,286,2
				"OliveOil",
				"FordA",
				"FordB",
                                "SonyAIBORobotSurface",
				"StarLightCurves",
				"Symbols",
				"TwoLeadECG"
		};
		

    
    
/* ARMA MODELS
 * Model for Y=log(lynx )-2.9036 is
y(t)=1.13y(t-1)-0.51y(t-2)+0.23y(t-3)-0.29y(t-4)+0.14y(t-5)-0.14y(t-6)+0.08y
(t-7)-0.04y(y-8)+0.13y(t-9)+0.19y(t-10)-0.31y(t-11)+e
 * from H. Tong,   "Some comments on the Canadian lynx data-with discussion"
J. Roy. Statist. Soc. A, 140  (1977)  pp. 432-435; 448-468
 */
public static double[][][] exampleModels={
	{	{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004},
		{1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}	},
			{	{0.488,	3.2496,	-1.6615,	-4.5163,	2.4227,	3.5192,	-1.9921,	-1.69,	1.0227,	0.5156,	-0.3423,	-0.0984,	0.0756,	0.0108,	-0.0109,	-0.0005,	0.001,	0,	-0.0001},
				{0.2212,	3.1268,	-0.7433,	-4.2211,	1.0535,	3.2423,	-0.8344,	-1.5702,	0.4107,	0.5012,	-0.1316,	-0.1069,	0.0279,	0.0151,	-0.0039,	-0.0014,	0.0003,	0.0001,	0}	},
	
			{	{-3.598,	-4.1991,	-0.3087,	3.2337,	2.5102,	0.3777,	-0.3093,	-0.1241,	-0.0009,	0.0057,	0.0008},
				{-2.8419,	-2.1235,	0.9762,	1.9624,	0.6231,	-0.1995,	-0.1281,	-0.0056,	0.0055,	0.0009}					},	
	
//Model 2 0.605	0.84					
				{	{-1.1084,-0.097,0.1579,0.0478,0.0044,0.0001},
					{-1.0993,-0.0436,0.1601,0.0426,0.0036,0.0001}		}
				


};
	
	
	/**
	 * Evaluate: Given a classifier, each of the Full, Clipped and Histogram evaluated on 
	//Benchmark		1. Euclidean
					2. DTW
	//Full model	3. Durban Levinsen Recursions
	//compressed 	4. Clipped+Durban Levinsen Recursions				
	//compressed 	4. Histograms+Euclidean distance			
	//compressed 	5. Histograms+	Gower metric
	//compressed 	5. FFT+	euclidean
	//compressed 	5. FFT+	likelihood
	 */

/** 
 * Experiment 1: Pure test of concept: show that DTW no use for ARMA data
 * Start with AR models of fixed length, n=500, with 1000 cases.
 * model 1:  Ar1=0.5. Vary parameter difference with model 2 from 0.5 to 0
 * 
 */
	public static void Experiment1_AR1_Test(int nosFiles,OutFile of){
		of.writeLine("ModelNos,EuclideanRAW,DTW_RAW,EuclideanARMA");
		Instances test,train,testARMA=null,trainARMA=null;
		
		for(int i=1;i<=nosFiles;i++){
//Load up the data
			System.out.println("Model"+i+",");
			String str="";
			of.writeString("Model"+i+",");
			test=ClassifierTools.loadData(
					SimulateAR.path+"AR1\\trainModel"+i);
			train=ClassifierTools.loadData(
					SimulateAR.path+"AR1\\testModel"+i);
//Train 1-NN, 1-NN DTW, on raw data and fitted ARMA data
			int nosClassifiers=2;
			Classifier[] all=new Classifier[nosClassifiers];
			NormalizableDistance df =new EuclideanDistance();
			df.setDontNormalize(true);
			all[0] = new kNN(df);
			all[1]=new kNN(new DTW_DistanceEfficient());
			ARMA ar=new ARMA();
			try{
				trainARMA=ar.process(train);
				testARMA=ar.process(test);
			}
			catch(Exception e){
				System.out.println("Error in transforming to arma"+e);
				System.exit(0);
			}
			str+=ClassifierTools.singleTrainTestSplitAccuracy(all[0],train,test)+",";
			str+=ClassifierTools.singleTrainTestSplitAccuracy(all[1],train,test)+",";
			str+=ClassifierTools.singleTrainTestSplitAccuracy(all[0],trainARMA,testARMA);
			System.out.println("RESULT ="+str);
			of.writeLine(str);
		}
	
	}

	
/** 
 * Experiment 2: Show histogram tends towards optimal as n increases for fixed AR1
 *  
 * Generate on the fly
 */
	public static void Experiment2_AR1_Classification(OutFile of){
		int startN=100;
		int endN=5000;
		int increment=200;
		int nosCases=100;
		int reps=10;
//		double[][] paras={{0.5,},{0.7}};
		double[][] paras={{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004},
		{1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}	};	

		IB1_Classification(paras,startN,endN,increment,nosCases,reps,of);
	}
	
/* Experiment 3: repeat 2 with more complex models, then introduce Jan's distance metric
 * 
 * 
 */
	public static void Experiment3_RandomAR_Classification(String fileName){
		
		int startN=100;
		int endN=1000;
		int increment=100;
		int nosCases=50;
		int reps=1;
		int modelReps=10;
		double[][] paras;
		OutFile of;
		for(int i=0;i<modelReps;i++){
			of=new OutFile(fileName+i+".csv");
			//Generate two random AR models
				//Random length between 4 and 15
			int nosParas=4+(int)(Math.random()*10);
			
			paras=new double[2][nosParas];
			for(int j=0;j<nosParas;j++){
				paras[0][j]=-0.8+1.9*Math.random();
				paras[1][j]=paras[0][j]-0.2+0.4*Math.random();
				if(paras[1][j]<=-1 || paras[1][j]>=1)
					paras[1][j]=-0.8+1.9*Math.random();
			}
			paras[0]=SimulateAR.findCoefficients(paras[0]);
			paras[1]=SimulateAR.findCoefficients(paras[1]);
			for(int j=0;j<nosParas;j++)
				of.writeString(paras[0][j]+",");
			of.writeString("\n");
			for(int j=0;j<nosParas;j++)
				of.writeString(paras[1][j]+",");
			of.writeString("\n");
			System.out.println("ARMA Model ="+nosParas);
			System.out.print("\n");
			DecimalFormat dc=new DecimalFormat("##.####");
			for(int j=0;j<nosParas;j++)
				System.out.print(dc.format(paras[0][j])+",");
			System.out.print("\n");
			for(int j=0;j<nosParas;j++)
				System.out.print(dc.format(paras[1][j])+",");
			
			System.out.print("\n");
				//Random parameters between -0.5 and 0.5
				//Measure histogram accuracy
					
			IB1_Classification(paras,startN,endN,increment,nosCases,reps,of);
			
			
		}
		
	}

	
	public static void Experiment4_TSDM_Problems(String fileName){
//Run ARMA, RL and DTW on standard TSDM problems
		Instances train,test;
		Instances armaTrain,armaTest;
		Instances histoTrain,histoTest;
//		Instances fftTest,fftTrain;
		OutFile of =new OutFile(fileName);
		of.writeLine("file,euclid,dtw,arma,histo");
		double euclidAcc=0,dtwAcc=0,histAcc=0,armaAcc=0;
		for(int i=0;i<fileNames.length;i++){
			System.out.println(" Running data set "+fileNames[i]);
	//1. Load test/train split
			String base=path+fileNames[i]+"\\"+fileNames[i];
			test=ClassifierTools.loadData(base+"_TEST");
			train=ClassifierTools.loadData(base+"_TRAIN");
	//2. Transform FFT, ARMA and Run Lengths
			ARMA ar=new ARMA();
			RunLength rl=new RunLength();
			int n=train.numAttributes()-1;
			rl.noGlobalMean();
			rl.setMaxRL(n/2);
			ar.setMaxLag(n/2);
			//3. Perform 4 accuracy measurements
			try{
				armaTrain=ar.process(train);
				armaTest=ar.process(test);
				histoTrain=rl.process(train);
				histoTest=rl.process(test);
				euclidAcc=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),train,test);
				Classifier c=new kNN(new DTW_DistanceBasic());
				dtwAcc=ClassifierTools.singleTrainTestSplitAccuracy(c,train,test);
				histAcc=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),histoTrain,histoTest);
				armaAcc=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),armaTrain,armaTest);
				
			}catch(Exception e){
				System.out.println("Error in process e = "+e);
				e.printStackTrace();
				System.exit(0);
			}
			//4. Write to file
			of.writeLine(fileNames[i]+","+euclidAcc+","+dtwAcc+","+armaAcc+","+histAcc);
		}
		
	}

	
	
	//Experiment 5: Show for a single model the relative accuracy of ARMA, RunLengths and FFT, DTW 	
//1. Generate the data sets
//2. For n=100 to 1000 
//	2. Measure accuracy and store. 	
	public static void Experiment5_AR_NearestNeighbour_SingleSeriesComparison(String fileName){
	//Generate a model
		int startN=100, endN=5100, increment=200;
		OutFile of=new OutFile(fileName);
		of.writeString("ARMA_Euclid,RL_Euclid,FFT,RL_Gower,RL_DTW\n");
		System.out.print("ARMA,RL_Euclid,RL_Gower,RL_Likelihood,RL_DTW\n");
		double[][] paras=exampleModels[0];
//		{{0.5},{0.7}};

//		{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004}
//		{1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}	

/*		Random r = new Random();
		r.setSeed(RANDOMSEED);
		
		int nosParas=3;
		double[][] paras=new double[2][];
		paras[0]=new double[nosParas];
		paras[1]=new double[nosParas];
		for(int j=0;j<nosParas;j++){
			paras[0][j]=-0.5+1*r.nextDouble();
			paras[1][j]=paras[0][j]-0.5+r.nextDouble();
			if(paras[1][j]<=-1)
				paras[1][j]=-.95;
			if(paras[1][j]>=1)
				paras[1][j]=0.95;
		}
*/	

		for(int m=0;m<1;m++)
		{
//			paras=generateValidModel(10,20);
			for(int n=startN;n<=endN;n+=increment){
				double[] acc=AR_NN_Classification(paras,n,100);
				of.writeString(m+","+n+",");
				System.out.print(m+","+n+",");
				for(int i=0;i<acc.length;i++){
					of.writeString(acc[i]+",");
					System.out.print(acc[i]+",");
				}
				of.writeString("\n");
				System.out.print("\n");
			}
		}
	}
	public static void Experiment6_AR_NN_DistanceMetricComparison(String fileName)
//This method generates the results for the section \subsection{Alternative Distance Measures for Run Lengths}
//Rerun with alternative n, manually hacked!	
	{

		
		//Generate a model
			int n=5000;
			int nosModels=200;
			OutFile of=new OutFile(fileName);
			of.writeString("ARMA_Euclid,RL_Euclid,FFT,RL_Gower,RL_DTW\n");
			System.out.print("ARMA,RL_Euclid,RL_Gower,RL_Likelihood,RL_DTW\n");
			double[][] paras;
			for(int m=0;m<nosModels;m++)
			{
				paras=generateValidModel(10,20);
				double[] acc=AR_NN_Classification(paras,n,100);
				of.writeString(m+","+n+",");
				System.out.print(m+","+n+",");
				for(int i=0;i<acc.length;i++){
					of.writeString(acc[i]+",");
					System.out.print(acc[i]+",");
				}
				of.writeString("\n");
				System.out.print("\n");
			}
		}
	public static void Experiment7_AR_RLvsFFT_AFC(String fileName){
		//Generate a model
			int startN=2000,endN=5000,inc=400;
			int nosModels=20;
			OutFile of=new OutFile(fileName);
			
			of.writeString("ARMA_Euclid,RL_DTW,FFT_Euclid,FFT_DTW,AFC_Euclid,AFC_DTW\n");
			System.out.print("ARMA_Euclid,RL_DTW,FFT_Euclid,FFT_DTW,AFC_Euclid,AFC_DTW\n");
			double[][] paras;
			for(int n=startN;n<endN;n+=inc){
				double[] av=new double[8];
				System.out.println("\n Running length = "+n+",");
				for(int m=0;m<nosModels;m++)
				{
					paras=generateValidModel(10,20);
					double[] acc=AR_TransformTest(paras,n,100);
					for(int i=0;i<acc.length;i++)
						av[i]+=acc[i];
					for(int i=0;i<acc.length;i++)
						System.out.print(acc[i]+",");
					System.out.print("\n");
					
				}
				of.writeString(nosModels+","+n+",");
				System.out.print(nosModels+","+n+",");
				for(int i=0;i<av.length;i++){
					of.writeString(av[i]/nosModels+",");
					System.out.print(av[i]/nosModels+",");
				}
				of.writeString("\n");
				System.out.print("\n");
			}
		}

	public static void Experiment8_AlternativeClassifiers(String fileName){
		//Generate a model
			int startN=2200,endN=5000,inc=400;
			int nosModels=30;
			int runs=30;
			OutFile of=new OutFile(fileName);
			
			of.writeString("ARMA_LDA,RL_LDA,FFT_LDA,ACF_LDA\n");
			System.out.print("ARMA_LDA,RL_LDA,FFT_LDA,ACF_LDA\n");
			double[][] paras;
			for(int n=startN;n<=endN;n+=inc){
				double[] av=new double[15];
				System.out.println("\n Running length = "+n+",");
				for(int m=0;m<nosModels;m++)
				{
					paras=generateValidModel(10,20);
					double[] acc;
					//1. Generate a random stationary model
					acc=AR_Mixed_Classification(paras,n,100);

					for(int j=0;j<av.length;j++)
						av[j]+=acc[j];
					for(int i=0;i<acc.length;i++)
						System.out.print(acc[i]+",");
					System.out.print("\n");
					
				}
				of.writeString(nosModels+","+n+",");
				System.out.print(nosModels+","+n+",");
				for(int i=0;i<av.length;i++){
					of.writeString(av[i]/nosModels+",");
					System.out.print(av[i]/nosModels+",");
				}
				of.writeString("\n");
				System.out.print("\n");
			}
		}

	public static void Experiment9_VariableLength(String fileName){
		//Fitting times
		int nosModels=20;
		double[][] paras;
		int smallN=500;
		int largeN=10000;
		int inc=500;
		OutFile of=new OutFile(fileName);
		for(int n=1000;n<largeN;n+=inc){
			double[] av=new double[3];
			System.out.println("\n Running length = "+n+",");
			for(int m=0;m<nosModels;m++)
			{
				paras=generateValidModel(10,20);
				double[] acc;
				//1. Generate a random stationary model
				acc=AR_FixedLength_Classification(paras,smallN,n,100);

				for(int j=0;j<av.length;j++)
					av[j]+=acc[j];
				System.out.print("run ="+m+",");
				for(int i=0;i<acc.length;i++)
					System.out.print(acc[i]+",");
				System.out.print("\n");
				
			}
			of.writeString(nosModels+","+n+",");
			System.out.print(nosModels+","+n+",");
			for(int i=0;i<av.length;i++){
				of.writeString(av[i]/nosModels+",");
				System.out.print(av[i]/nosModels+",");
			}
			of.writeString("\n");
			System.out.print("\n");
		
		}
	}
	public static void Experiment10_Timing(String fileName){
		int nosModels=1;
		double[][] paras;
		int startN=4500;
		int endN=10000;
		int inc=500;
		OutFile of=new OutFile(fileName);
		for(int n=startN;n<endN;n+=inc){
			double[] av=new double[3];
			System.out.println("\n Running length = "+n+",");
			for(int m=0;m<nosModels;m++)
			{
				paras=generateModel(10,20);
				double[] acc;
				//1. Generate a random stationary model
				acc=AR_TimingExperiment(paras,n,100);

				for(int j=0;j<av.length;j++)
					av[j]+=acc[j]/(double)1000000;
				System.out.print("run ="+m+",");
				for(int i=0;i<acc.length;i++)
					System.out.print(acc[i]/1000000+",");
				System.out.print("\n");
				
			}
			of.writeString(nosModels+","+n+",");
			System.out.print(nosModels+","+n+",");
			for(int i=0;i<av.length;i++){
				of.writeString(av[i]/nosModels+",");
				System.out.print(av[i]/nosModels+",");
			}
			of.writeString("\n");
			System.out.print("\n");
		
			}
		}
		public static void Experiment11_CrossPoints(String fileName){
			int nosModels=10;
			double[][] paras;
			int startN=1000;
			int endN=5000;
			int inc=200;
			OutFile of=new OutFile(fileName);
			for(int n=startN;n<=endN;n+=inc){
				System.out.println("\n Running base length = "+n+",");
				boolean beat=false;
				int longN=n+2*inc;	
				while(!beat&& longN<5000){
					double[] av=new double[3];					
					for(int m=0;m<nosModels;m++){
						paras=generateValidModel(10,20);
						double[] acc;
						acc=AR_FixedLength_Classification(paras,n,longN,100);
	
						for(int j=0;j<av.length;j++)
							av[j]+=acc[j];
						System.out.print("run ="+m+","+" n ="+n+" longN="+longN+",");
						for(int i=0;i<acc.length;i++)
							System.out.print(acc[i]+",");
						System.out.print("\n");
					
					}
					if(av[1]>=av[0] && av[1]>=av[2]){
						beat=true;
						of.writeString(nosModels+","+n+","+","+longN+",");
						System.out.print(nosModels+","+n+","+","+longN+",");
						for(int i=0;i<av.length;i++){
							of.writeString(av[i]/nosModels+",");
							System.out.print(av[i]/nosModels+",");
						}
						of.writeString("\n");
						System.out.print("\n");
					}
					else
						longN+=inc;	
			}
		}	
	}
	
	
	public static int RANDOMSEED=7;
	public static void main(String[] args){
//		Experiment1_AR1_Test(50, new OutFile("C:\\Research\\Results\\RunLengths\\Experiment1.csv"));
//		Experiment2_AR1_Classification(new OutFile("C:\\Research\\Results\\RunLengths\\Experiment2.csv"));
//		Experiment3_RandomAR_Classification("C:\\Research\\Results\\RunLengths\\Experiment3_");
		Experiment4_TSDM_Problems("C:\\Research\\Results\\RunLengths\\Experiment4_TSDMProblems.csv");	
//		Experiment5_AR_NearestNeighbour_SingleSeriesComparison("C:\\Research\\Results\\RunLengths\\Experiment5_SingleModelComparison.csv");
//		Experiment6_AR_NN_DistanceMetricComparison("C:\\Research\\Results\\RunLengths\\Experiment6_DistanceMetricComparison4.csv");
//		Experiment7_AR_RLvsFFT_AFC("C:\\Research\\Results\\RunLengths\\Experiment7_RLvsFFT_AFC.csv");
//		Experiment8_AlternativeClassifiers("C:\\Research\\Results\\RunLengths\\Experiment8_DifferentClassifiers.csv");
//		Experiment7_AR_VaryingSize_MultipleSeriesComparison("C:\\Research\\Results\\RunLengths\\Experiment7_MultipleClassifierComparison2.csv");
//		Experiment9_VariableLength("C:\\Research\\Results\\RunLengths\\Experiment9_DiffentN.csv");
//		Experiment10_Timing("C:\\Research\\Results\\RunLengths\\Timings.csv");
//		Experiment11_CrossPoints("C:\\Research\\Results\\RunLengths\\CrossPoints.csv");
//		Experiment6_AIC_Lengths("C:\\Research\\Results\\RunLengths\\AIC_Lengths.csv");
//		Experiment7_AIC_Effect("C:\\Research\\Results\\RunLengths\\AIC_Effect.csv");

		
//		FFT_Test("C:\\Research\\Results\\RunLengths\\FFT_Test.csv");		
//		exampleSeries("C:\\Research\\Results\\RunLengths\\exampleSeries2.csv");
//		generateValidModels("C:\\Research\\Results\\RunLengths\\ValidModels.csv");
		f1HistogramsBasic();
	}
/**
 * This method randomly generates 100 stationary arma models for use in classification experiments. It
 * 1. Randomly generate model length between 3 and 10
 * 2. Randomly generate first model parameters
 * 3. Perturb each parameter by 10% for second model
 * 4. Call SimulateAR.findCoefficients to find AR paras
 * 5. Generate a test samples of 100 and 1000
 * 6. Test if zero mean (series mean between -0.5 and 0.5 
 * 7. Measure accuracy with ARMA fit. Must be above 60% for 100 and above 90% for 1000
 * 
 * @param fileName
 */
	
	
	public static double[][] generateValidModel(int minLength, int maxLength){
		double[][] paras=new double[2][];
		Random r = new Random();
		r.setSeed(RANDOMSEED);
		int[] cases={100,100};
		boolean good=false;
		while(!good){
			good=true;
			int nosParas1=(int)(minLength+maxLength*r.nextDouble());
			int nosParas2=(int)(nosParas1+(2-4*r.nextDouble()));
			 paras[0]=new double[nosParas1];
			paras[1]=new double[nosParas2];
			for(int j=0;j<nosParas1;j++){
				paras[0][j]=-0.9+1.8*r.nextDouble();
			}
			for(int j=0;j<nosParas2;j++){
				if(j<nosParas1){
					if(r.nextDouble()>0.5)
						paras[1][j]=paras[0][j]*(1+0.1*r.nextDouble());
					else
						paras[1][j]=paras[0][j]*(1-0.1*r.nextDouble());
					if(paras[1][j]<=-1)
						paras[1][j]=-.95;
					if(paras[1][j]>=1)
						paras[1][j]=0.95;
				}
			}
			// 4. Call SimulateAR.findCoefficients to find AR paras
			paras[0]=SimulateAR.findCoefficients(paras[0]);
			paras[1]=SimulateAR.findCoefficients(paras[1]);	
// 5. Generate a test samples of 500 and 1000
			Instances smallTrain =SimulateAR.generateARDataSet(paras,400,cases);
			Instances smallTest =SimulateAR.generateARDataSet(paras,400,cases);
			Instances largeTrain =SimulateAR.generateARDataSet(paras,1000,cases);
			Instances largeTest =SimulateAR.generateARDataSet(paras,1000,cases);
// 6. Test if zero mean (series mean between -0.5 and 0.5 
			if(zeroMeans(smallTrain)||zeroMeans(smallTest)||zeroMeans(largeTrain)||zeroMeans(largeTest)){
				good=false;
			}
			ARMA ar =new ARMA();
			double smallAcc=0,largeAcc=0;
			try{
				Instances arTrain=ar.process(smallTrain);
				Instances arTest=ar.process(smallTest);
				 smallAcc=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),arTrain,arTest);
				if(smallAcc<0.6||smallAcc>0.9){
					good=false;
				}
			}catch(Exception e){
				System.out.println("Exception in transformation! skipping");
				good=false;
			}
			try{
				Instances arTrain=ar.process(largeTrain);
				Instances arTest=ar.process(largeTest);
				largeAcc=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),arTrain,arTest);
				if(largeAcc<=smallAcc|| largeAcc==1.0){
					System.out.println("Rejecting model for being too hard or easy on the LARGE set. Acc ="+largeAcc);
					good=false;
				}
			}catch(Exception e){
				System.out.println("Exception in transformation! skipping");
				good=false;
			}
			if(good)
				return paras;
				
		}
		return null;
	}
	public static double[][] generateModel(int minLength, int maxLength){
		double[][] paras=new double[2][];
		Random r = new Random();
		r.setSeed(RANDOMSEED);
		int[] cases={100,100};
		boolean good=false;
		while(!good){
			good=true;
			int nosParas1=(int)(minLength+maxLength*r.nextDouble());
			int nosParas2=(int)(nosParas1+(2-4*r.nextDouble()));
			 paras[0]=new double[nosParas1];
			paras[1]=new double[nosParas2];
			for(int j=0;j<nosParas1;j++){
				paras[0][j]=-0.9+1.8*r.nextDouble();
			}
			for(int j=0;j<nosParas2;j++){
				if(j<nosParas1){
					if(r.nextDouble()>0.5)
						paras[1][j]=paras[0][j]*(1+0.1*r.nextDouble());
					else
						paras[1][j]=paras[0][j]*(1-0.1*r.nextDouble());
					if(paras[1][j]<=-1)
						paras[1][j]=-.95;
					if(paras[1][j]>=1)
						paras[1][j]=0.95;
				}
			}
			// 4. Call SimulateAR.findCoefficients to find AR paras
			paras[0]=SimulateAR.findCoefficients(paras[0]);
			paras[1]=SimulateAR.findCoefficients(paras[1]);	
// 5. Generate a test samples of 500 and 1000
			return paras;
				
		}
		return null;
	}
	public static void generateValidModels(String fileName){
		int modelCount=0;
		int nosParas1,nosParas2;
		double[][] paras=new double[2][];
		Random r = new Random();
		r.setSeed(RANDOMSEED);
		int[] cases={100,100};
		boolean good=true;
		DecimalFormat dc = new DecimalFormat("###.####");
		OutFile of = new OutFile(fileName);
		
		while(modelCount<100){
			good=true;
// 1. Randomly generate model length between 2 and 10
			nosParas1=(int)(10+20*r.nextDouble());
			nosParas2=(int)(nosParas1+(2-4*r.nextDouble()));
// 2. Randomly generate first model parameters
// 3. Perturb each parameter by 10% for second model

			 paras[0]=new double[nosParas1];
			paras[1]=new double[nosParas2];
			for(int j=0;j<nosParas1;j++){
				paras[0][j]=-0.9+1.8*r.nextDouble();
			}
			for(int j=0;j<nosParas2;j++){
				if(j<nosParas1){
					if(r.nextDouble()>0.5)
						paras[1][j]=paras[0][j]*(1+0.1*r.nextDouble());
					else
						paras[1][j]=paras[0][j]*(1-0.1*r.nextDouble());
					if(paras[1][j]<=-1)
						paras[1][j]=-.95;
					if(paras[1][j]>=1)
						paras[1][j]=0.95;
				}
			}
			

// 4. Call SimulateAR.findCoefficients to find AR paras
			paras[0]=SimulateAR.findCoefficients(paras[0]);
			paras[1]=SimulateAR.findCoefficients(paras[1]);	
// 5. Generate a test samples of 500 and 1000
			Instances smallTrain =SimulateAR.generateARDataSet(paras,100,cases);
			Instances smallTest =SimulateAR.generateARDataSet(paras,100,cases);
			Instances largeTrain =SimulateAR.generateARDataSet(paras,1000,cases);
			Instances largeTest =SimulateAR.generateARDataSet(paras,1000,cases);
// 6. Test if zero mean (series mean between -0.5 and 0.5 
			if(zeroMeans(smallTrain)||zeroMeans(smallTest)||zeroMeans(largeTrain)||zeroMeans(largeTest)){
				System.out.println("Rejecting model for non zero means");
				System.out.println("Small data are"+smallTrain);
				good=false;
			}
			
// 7. Measure accuracy with ARMA fit. Must be above 60% for 100 and above 90% for 1000
			ARMA ar =new ARMA();
			double smallAcc=0,largeAcc=0;
			try{
				Instances arTrain=ar.process(smallTrain);
				Instances arTest=ar.process(smallTest);
				 smallAcc=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),arTrain,arTest);
				if(smallAcc<0.6||smallAcc>0.9){
					System.out.println("Rejecting model for being too hard or easy on the small set. Acc ="+smallAcc);
					good=false;
				}
			}catch(Exception e){
				System.out.println("Exception in transformation! skipping");
				good=false;
			}
			try{
				Instances arTrain=ar.process(largeTrain);
				Instances arTest=ar.process(largeTest);
				largeAcc=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),arTrain,arTest);
				if(largeAcc<=smallAcc|| largeAcc==1.0){
					System.out.println("Rejecting model for being too hard or easy on the LARGE set. Acc ="+largeAcc);
					good=false;
				}
			}catch(Exception e){
				System.out.println("Exception in transformation! skipping");
				good=false;
			}
			
			
			if(good==true){
				System.out.println("KEEPING MODEL >");
				of.writeLine(smallAcc+","+largeAcc);
				of.writeString(paras[0].length+",");
				for(int i=0;i<paras[0].length;i++){
					of.writeString(dc.format(paras[0][i])+",");
					System.out.print(dc.format(paras[0][i])+",");
				}
				of.writeString("\n");
				System.out.print("\n");
				of.writeString(paras[1].length+",");
				for(int i=0;i<paras[1].length;i++){
					of.writeString(dc.format(paras[1][i])+",");
					System.out.print(dc.format(paras[1][i])+",");
				}
				of.writeLine("\n");
				System.out.println("\nACC Small = "+smallAcc+" ACC Large ="+largeAcc);
				System.out.print("\n\n");
				modelCount++;
			}
		}
	}
	public static boolean zeroMeans(Instances d){
		for(int i=0;i<d.numInstances();i++){
			double mean=0;
			int count=0;
			Instance inst=d.instance(i);
			for(int j=0;j<inst.numAttributes();j++){
				if(j!=inst.classIndex()){
					mean+=inst.value(j);
					count++;
				}
				mean/=count;
				if(mean<-0.5|| mean>0.5) return false;
			}
		}
		return true;
			
	}
	
	public static double[] AR_NN_Classification(double[][] paras, int n,int nosCases){
		double[] acc=new double[5];
		Instances train,test;
		Instances arTrain,arTest;
		Instances rlTrain,rlTest;
		Instances fftTrain,fftTest;
		
		int[] cases={nosCases,nosCases};
		//1. Generate a random stationary model
		train=SimulateAR.generateARDataSet(paras,n,cases);
		test=SimulateAR.generateARDataSet(paras,n,cases);

		//2. Transform to ARMA, RunLengths and FFT.
		
		ARMA ar=new ARMA();
		RunLength rl=new RunLength();
		FFT fft=new FFT();
		//Go for 10% compression
		rl.setMaxRL(n/10);
		ar.setMaxLag(n/10);
		ar.setUseAIC(true);
		
		fft.padSeries(true);
		try{
			arTrain=ar.process(train);
			arTest=ar.process(test);
			rlTrain=rl.process(train);
			rlTest=rl.process(test);
			fftTrain=fft.process(train);
			fftTest=fft.process(test);
			fft.truncate(fftTrain,n/10);
			fft.truncate(fftTest,n/10);

			//3. Do test train accuracy with.
			Classifier dtw,gower;
			dtw=new kNN(new DTW_DistanceBasic());
			gower=new kNN(new GowerDistance(rlTrain));
			
			acc[0]=ClassifierTools.singleTrainTestSplitAccuracy(new kNN(new EuclideanDistance()),arTrain,arTest);
			acc[1]=ClassifierTools.singleTrainTestSplitAccuracy(new kNN(new EuclideanDistance()),rlTrain,rlTest);
			acc[2]=ClassifierTools.singleTrainTestSplitAccuracy(new kNN(new GowerDistance(rlTrain)),rlTrain,rlTest);
			acc[3]=ClassifierTools.singleTrainTestSplitAccuracy(new kNN(new LikelihoodRatioDistance()),rlTrain,rlTest);
			acc[4]=ClassifierTools.singleTrainTestSplitAccuracy(dtw,rlTrain,rlTest);
			
		}catch(Exception e){
			System.out.println("Error w ="+e);
			System.exit(0);
		}
		return acc;
	}

	public static double[] AR_TransformTest(double[][] paras, int n,int nosCases){
		double[] acc=new double[4];
		Instances train,test;
		Instances arTrain,arTest;
		Instances rlTrain,rlTest;
		Instances fftTrain,fftTest;
		Instances acfTrain,acfTest;
		
		int[] cases={nosCases,nosCases};
		//1. Generate a random stationary model
		train=SimulateAR.generateARDataSet(paras,n,cases);
		test=SimulateAR.generateARDataSet(paras,n,cases);

		//2. Transform to ARMA, RunLengths and FFT.
		
		ARMA ar=new ARMA();
		RunLength rl=new RunLength();
		FFT fft=new FFT();
		ACF acf= new ACF();
		
		//Go for 10% compression
		rl.setMaxRL(n/10);
		ar.setMaxLag(n/10);
		ar.setUseAIC(true);
		acf.setMaxLag(n/10);
		fft.padSeries(true);
		try{
			arTrain=ar.process(train);
			arTest=ar.process(test);
			rlTrain=rl.process(train);
			rlTest=rl.process(test);
			fftTrain=fft.process(train);
			fftTest=fft.process(test);
			acfTrain=acf.process(train);
			acfTest=acf.process(test);
			
			fft.truncate(fftTrain,n/10);
			fft.truncate(fftTest,n/10);
			//3. Do test train accuracy with.
			
			acc[0]=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),arTrain,arTest);
			acc[1]=ClassifierTools.singleTrainTestSplitAccuracy(new kNN(new DTW_DistanceBasic()),rlTrain,rlTest);
			acc[2]=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),fftTrain,fftTest);
			acc[3]=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),acfTrain,acfTest);

			
		}catch(Exception e){
			System.out.println("Error w ="+e);
			e.printStackTrace();
			System.exit(0);
		}
		return acc;
	}	
	
//1-NN, C4.5, Naive Bayes, RandomForests with AR, RL and FFT at 10%	
	public static double[] AR_Mixed_Classification(double[][] paras, int n,int nosCases){
		double[] acc=new double[15];
		Instances train,test;
		Instances arTrain,arTest;
		Instances rlTrain,rlTest;
		Instances fftTrain,fftTest;
		Instances acfTrain,acfTest;
		
		int[] cases={nosCases,nosCases};
		//1. Generate a random stationary model
		train=SimulateAR.generateARDataSet(paras,n,cases);
		test=SimulateAR.generateARDataSet(paras,n,cases);

		//2. Transform to ARMA, RunLengths and FFT.
		
		ARMA ar=new ARMA();
		RunLength rl=new RunLength();
		FFT fft=new FFT();
		ACF acf= new ACF();
		
		//Go for 10% compression
		rl.setMaxRL(n/10);
		ar.setMaxLag(n/10);
		ar.setUseAIC(true);
		acf.setMaxLag(n/10);
		fft.padSeries(true);
		try{
			arTrain=ar.process(train);
			arTest=ar.process(test);
			rlTrain=rl.process(train);
			rlTest=rl.process(test);
			fftTrain=fft.process(train);
			fftTest=fft.process(test);
			acfTrain=acf.process(train);
			acfTest=acf.process(test);
			
			fft.truncate(fftTrain,n/10);
			fft.truncate(fftTest,n/10);
			//3. Do test train accuracy with.
			Classifier dtw;
			dtw=new kNN(new DTW_DistanceBasic());
			
			acc[0]=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),arTrain,arTest);
			acc[1]=ClassifierTools.singleTrainTestSplitAccuracy(new J48(),arTrain,arTest);
			acc[2]=ClassifierTools.singleTrainTestSplitAccuracy(new NaiveBayes(),arTrain,arTest);
			acc[3]=ClassifierTools.singleTrainTestSplitAccuracy(new RandomForest(),arTrain,arTest);
			acc[4]=ClassifierTools.singleTrainTestSplitAccuracy(new Logistic(),arTrain,arTest);
			acc[5]=ClassifierTools.singleTrainTestSplitAccuracy(dtw,rlTrain,rlTest);
			acc[6]=ClassifierTools.singleTrainTestSplitAccuracy(new J48(),rlTrain,rlTest);
			acc[7]=ClassifierTools.singleTrainTestSplitAccuracy(new NaiveBayes(),rlTrain,rlTest);
			acc[8]=ClassifierTools.singleTrainTestSplitAccuracy(new RandomForest(),rlTrain,rlTest);
			acc[9]=ClassifierTools.singleTrainTestSplitAccuracy(new Logistic(),rlTrain,rlTest);
			acc[10]=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),acfTrain,acfTest);
			acc[11]=ClassifierTools.singleTrainTestSplitAccuracy(new J48(),acfTrain,acfTest);
			acc[12]=ClassifierTools.singleTrainTestSplitAccuracy(new NaiveBayes(),acfTrain,acfTest);
			acc[13]=ClassifierTools.singleTrainTestSplitAccuracy(new RandomForest(),acfTrain,acfTest);
			acc[14]=ClassifierTools.singleTrainTestSplitAccuracy(new Logistic(),acfTrain,acfTest);
			
		}catch(Exception e){
			System.out.println("Error w ="+e);
			e.printStackTrace();
			System.exit(0);
		}
		return acc;
	}	

	//1-NN, C4.5, Naive Bayes, RandomForests with AR, RL and FFT at 10%	
	public static double[] AR_TimingExperiment(double[][] paras, int n, int nosCases){
		double[] acc=new double[3];
		Instances train,test;
		Instances arTrain,arTest;
		Instances rlTrain,rlTest;
		Instances fftTrain,fftTest;
		Instances acfTrain,acfTest;
		int reps=30;
		
		int[] cases={nosCases,nosCases};
		//1. Generate a random stationary model
		train=SimulateAR.generateARDataSet(paras,n,cases);
		try{
			//2. Transform to RL
			RunLength rl=new RunLength();
			ARMA ar=new ARMA();
			ACF acf= new ACF();
		//Go for 10% compression
			ar.setMaxLag(n/10);
			ar.setUseAIC(true);
			acf.setMaxLag(n/10);
			rl.setMaxRL(n/10);
			long start=System.nanoTime();
			for(int i=0;i<reps;i++)
				rlTrain=rl.process(train);
			start=System.nanoTime()-start;
			acc[0]+=(double)start/(double)reps;
		//ARMA and ACF		
			start=System.nanoTime();
			for(int i=0;i<reps;i++)
				arTrain=ar.process(train);
			start=System.nanoTime()-start;
			acc[1]+=(double)start/(double)reps;
			start=System.nanoTime();
			for(int i=0;i<reps;i++)
				acfTrain=acf.process(train);
			start=System.nanoTime()-start;
			acc[2]+=(double)start/(double)reps;
		}catch(Exception e){
			System.out.println("Error w ="+e);
			e.printStackTrace();
			System.exit(0);
		}
		return acc;
	}	
	
	
	
	//
	public static double[] AR_FixedLength_Classification(double[][] paras, int shortN,int longN, int nosCases){
		double[] acc=new double[3];
		Instances train,test;
		Instances arTrain,arTest;
		Instances rlTrain,rlTest;
		Instances fftTrain,fftTest;
		Instances acfTrain,acfTest;
		
		int[] cases={nosCases,nosCases};
		//1. Generate a random stationary model
		train=SimulateAR.generateARDataSet(paras,longN,cases);
		test=SimulateAR.generateARDataSet(paras,longN,cases);
		try{
			//2. Transform to RL
			RunLength rl=new RunLength();
			rl.setMaxRL(shortN/10);
			rlTrain=rl.process(train);
			rlTest=rl.process(test);
		//3. Remove data down to short N
			for(int i=shortN;i<longN;i++){
				train.deleteAttributeAt(shortN-1);
				test.deleteAttributeAt(shortN-1);
			}
		//ARMA and ACF		
			ARMA ar=new ARMA();
			ACF acf= new ACF();
		//Go for 10% compression
			ar.setMaxLag(shortN/10);
			ar.setUseAIC(true);
			acf.setMaxLag(shortN/10);

			arTrain=ar.process(train);
			arTest=ar.process(test);
			acfTrain=acf.process(train);
			acfTest=acf.process(test);
			//3. Do test train accuracy with.
			Classifier dtw;
			dtw=new kNN(new DTW_DistanceBasic());
			acc[0]=ClassifierTools.singleTrainTestSplitAccuracy(new Logistic(),arTrain,arTest);
			acc[1]=ClassifierTools.singleTrainTestSplitAccuracy(dtw,rlTrain,rlTest);
			acc[2]=ClassifierTools.singleTrainTestSplitAccuracy(new Logistic(),acfTrain,acfTest);
			
		}catch(Exception e){
			System.out.println("Error w ="+e);
			e.printStackTrace();
			System.exit(0);
		}
		return acc;
	}	
	
	public static void IB1_Classification(double[][] paras, int startN, int endN, int increment, int nosCases, int reps, OutFile of){
		double euclidAcc=0,histAcc=0, armaAcc=0;
		Instances train,test;
		Instances armaTrain,armaTest;
		Instances histoTrain,histoTest;
		int[] cases={nosCases,nosCases};
		of.writeLine("n,euclid,histogram,arma");
		for(int n=startN;n<=endN;n+=increment){
			System.out.println(" Running with series length ="+n);
			histAcc=0;
			armaAcc=0;
			euclidAcc=0;
			of.writeString(n+",");
			for(int r=1;r<=reps;r++){
//1. Generate two class problem with nosCases in each class, each series length n
			train=SimulateAR.generateARDataSet(paras,n,cases);
			test=SimulateAR.generateARDataSet(paras,n,cases);
//2. transform to ARMA and Histogram
			ARMA ar=new ARMA();
			RunLength rl=new RunLength();
			rl.noGlobalMean();
			rl.setMaxRL(n/4);
			ar.setMaxLag(n/4);
			try{
				armaTrain=ar.process(train);
				armaTest=ar.process(test);
				histoTrain=rl.process(train);
				histoTest=rl.process(test);
				euclidAcc+=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),train,test);
				histAcc+=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),histoTrain,histoTest);
				armaAcc+=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),armaTrain,armaTest);
			}catch(Exception e){
				System.out.println("Error in process e = "+e);
				e.printStackTrace();
				System.exit(0);
			}
//3. Measure classification accuracy on both 	
			}
//			euclidAcc/=reps;
			histAcc/=reps;
			armaAcc/=reps;
			System.out.println("Euclid="+euclidAcc+" Histo ="+histAcc+" ARMA ="+armaAcc);
			of.writeLine(histAcc+","+armaAcc);
		}
		
	}
	
//This is to create similar looking series that differ in AR structure and in histograms	
	public static void exampleSeries(String fileName){
		int n=512;
		int nosParas=3;
		Random r = new Random();
		r.setSeed(RANDOMSEED);
		OutFile of=new OutFile(fileName);
		double[][] paras=new double[2][nosParas];
		for(int j=0;j<nosParas;j++){
			paras[0][j]=-0.5+1*r.nextDouble();
			paras[1][j]=paras[0][j]-0.5+r.nextDouble();
			if(paras[1][j]<=-1)
				paras[1][j]=-.95;
			if(paras[1][j]>=1)
				paras[1][j]=0.95;
		}
		paras[0]=SimulateAR.findCoefficients(paras[0]);
		paras[1]=SimulateAR.findCoefficients(paras[1]);
		for(int j=0;j<nosParas;j++)
			of.writeString("\n");		
		for(int j=0;j<nosParas;j++)
			of.writeString(paras[0][j]+",");
		of.writeString("\n");
		for(int j=0;j<nosParas;j++)
			of.writeString(paras[1][j]+",");
		of.writeString("\n");
		System.out.println("ARMA Model ="+nosParas);
		System.out.print("\n");
		DecimalFormat dc=new DecimalFormat("##.####");
		for(int j=0;j<nosParas;j++)
			System.out.print(dc.format(paras[0][j])+",");
		System.out.print("\n");
		for(int j=0;j<nosParas;j++)
			System.out.print(dc.format(paras[1][j])+",");
		System.out.print("\n");
//Generate two series of length 500
		int[] cases={2,2};
		Instances train=SimulateAR.generateARDataSet(paras,n,cases);
		double[][] data=new double[train.numInstances()][];
		for(int i=0;i<train.numInstances();i++)
			data[i]=train.instance(i).toDoubleArray();
		for(int i=0;i<data.length;i++)
			of.writeString(dc.format(data[i][data[i].length-1])+",");
		of.writeString("\n");
		for(int j=0;j<data[0].length-1;j++){
			for(int i=0;i<data.length;i++)
				of.writeString(dc.format(data[i][j])+",");
			of.writeString("\n");
		}
//Fit ARMA
		ARMA ar= new ARMA();
		ar.setUseAIC(true);
//Fit Histogram		
		RunLength rl=new RunLength();
		rl.noGlobalMean();
		rl.setMaxRL(n/4);
		ar.setMaxLag(n/4);
		try{
			Instances arTrain=ar.process(train);
			data=new double[train.numInstances()][];
			for(int i=0;i<arTrain.numInstances();i++)
				data[i]=arTrain.instance(i).toDoubleArray();
			System.out.print("\n ARMA FIT =\n");
			for(int i=0;i<arTrain.numInstances();i++){
				for(int j=0;j<30;j++){
					System.out.print(dc.format(data[i][j])+",");
				}
				System.out.print("\n");
			}
			System.out.print("\n");
			of.writeString("\n");
			of.writeString("\n");
			for(int i=0;i<data.length;i++)
				of.writeString(dc.format(data[i][data[i].length-1])+",");
			of.writeString("\n");
			System.out.print("\n RUN LENGTH FIT =\n");
			for(int j=0;j<data[0].length-1;j++){
				for(int i=0;i<data.length;i++)
					of.writeString(dc.format(data[i][j])+",");
				of.writeString("\n");
			}
			of.writeString("\n");
			of.writeString("\n");
			Instances histTrain=rl.process(train);
			data=new double[train.numInstances()][];
			for(int i=0;i<train.numInstances();i++)
				data[i]=histTrain.instance(i).toDoubleArray();

			for(int i=0;i<arTrain.numInstances();i++){
				for(int j=0;j<data[i].length;j++){
					System.out.print(dc.format(data[i][j])+",");
				}
				System.out.print("\n");
			}
			
			for(int i=0;i<data.length;i++)
				of.writeString(dc.format(data[i][data[i].length-1])+",");
			of.writeString("\n");
			for(int j=0;j<data[0].length-1;j++){
				for(int i=0;i<data.length;i++)
					of.writeString(dc.format(data[i][j])+",");
				of.writeString("\n");
			}
		}catch(Exception e){
			System.out.println("Exception = "+e);
			System.exit(0);
		}
	}
	public static void runLengthTest(){
		for(int i=0;i<fileNames.length;i++)
		{
			Instances train=ClassifierTools.loadData(path+fileNames[i]+"\\"+fileNames[i]+"_TRAIN");
			Instances test=ClassifierTools.loadData(path+fileNames[i]+"\\"+fileNames[i]+"_TEST");
			//Filter through RLE clipper
			RunLength rl=new RunLength();
			rl.noGlobalMean();
			Clipping clip=new Clipping();
			try{
				Classifier c=new IBk();
				System.out.print("\n"+fileNames[i]+"\t");
				double a = ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);
				System.out.print(a+"\t");
				c=new kNN(new DTW_DistanceBasic());
				a = ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);
				System.out.print(a+"\t");
				Instances clipTrain=clip.process(train);
				Instances clipTest=clip.process(test);
				a = ClassifierTools.singleTrainTestSplitAccuracy(c, clipTrain, clipTest);
				System.out.print(a+"\t");
				Instances rlTrain=rl.process(train);
				Instances rlTest=rl.process(test);
				a = ClassifierTools.singleTrainTestSplitAccuracy(c, rlTrain, rlTest);
				System.out.print(a+"\t");
				
			}catch(Exception e){
				System.exit(0);
			}
		}
	}

	//DEPRECIATED: NOT USED
	public static String SingleAR1_Experiment(String file){
		String path=SimulateAR.path+"AR1\\";
		String str="";
//Set up filters for data sets. Clipped data needs to be stored as reals to work with knn
		
		Clipping clip=new Clipping();
		clip.setUseRealAttributes(true);
////Max run length set, 
		RunLength rl=new RunLength();
		rl.noGlobalMean();
		rl.setMaxRL(10);
//ARMA fitted model with DL recursions, no AIC stopping I THINK! check. Max length of model
//25% of series length
		ARMA ar=new ARMA();
		
//FFT: keep only first 25% of terms?		
		FFT fft =new FFT();
		int nosDataSets=5;
		Instances[] train=new Instances[nosDataSets];
		Instances[] test=new Instances[nosDataSets];
		

		train[0]=ClassifierTools.loadData(path+file+"\\"+file+"_TRAIN");
		test[0]=ClassifierTools.loadData(path+file+"\\"+file+"_TEST");
		try{
			train[1]=clip.process(train[0]);
			test[1]=clip.process(test[0]);
//			System.out.println("Clipped Test ="+test[1]);
			train[2]=rl.process(train[1]);
			test[2]=rl.process(test[1]);
			train[3]= ar.process(train[0]);
			test[3]= ar.process(test[0]);
			train[3]= fft.process(train[0]);
			test[3]= fft.process(test[0]);
			System.out.println("ARMA Test ="+train[3]);
		}catch(Exception e){
			System.out.println("Exception in the filters ="+e);
			e.printStackTrace();
			System.exit(0);
		}
//		System.out.println("Train clipped"+train[1]);
//		System.out.println("Train histo"+train[2]);

		//For each type of data (RAW, CLIPPED and RL) try the following classifiers
		//1. 1-NN Euclid
		
//Raw data classifiers		
		int nosClassifiers=2;
		Classifier[] all=new Classifier[nosClassifiers];
		NormalizableDistance df =new EuclideanDistance();
		df.setDontNormalize(true);
		all[0] = new kNN(df);
		all[1]=new kNN(new DTW_DistanceEfficient());
		
/*			all[0] = new IBk(1);
		all[1]=new kNN(new DTW_DistanceEfficient());
	all[3]=new DTW_kNN();
		((DTW_kNN)all[3]).optimiseWindow(true);
*/
		
		for(int j=0;j<nosClassifiers;j++)
		for(int i=0;i<train.length;i++)
				str+=ClassifierTools.singleTrainTestSplitAccuracy(all[j],train[i],test[i])+",";

		//Raw data classifiers		

//Transform classifiers		
		
		//2/ Clipped
			//2.1 Euclidean

			//2.2 DTW
		
			//2.3 ARMA Model using DL recursions 
		
			//
		
		
		return str;
	}
//FFT Test
	
	
	public static void FFT_Test(String fileName){
		int startN=100;
		int endN=1000;
		int increment=200;
		int nosCases=2;
		int reps=10;
//		double[][] paras={{0.5},{0.7}};
		double[][] paras={{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004},
		{1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}	};	
		OutFile of=new OutFile(fileName);
		double fftAcc=0,histAcc=0, armaAcc=0;
		Instances train,test;
		Instances armaTrain,armaTest;
		Instances histoTrain,histoTest;
		Instances fftTrain,fftTest;
		int[] cases={nosCases,nosCases};
		
		of.writeLine("n,euclid,histogram,arma");
		for(int n=startN;n<=endN;n+=increment){
			System.out.println(" Running with series length ="+n);
			histAcc=0;
			armaAcc=0;
			fftAcc=0;
			of.writeString(n+",");
			for(int r=1;r<=reps;r++){
	//1. Generate two class problem with nosCases in each class, each series length n
				train=SimulateAR.generateARDataSet(paras,n,cases);
				test=SimulateAR.generateARDataSet(paras,n,cases);
				of.writeLine(train+"\n");
				//2. transform to ARMA and Histogram
				ARMA ar=new ARMA();
				RunLength rl=new RunLength();
				FFT fft=new FFT();

				rl.setMaxRL(n/4);
				ar.setMaxLag(n/4);
				fft.padSeries(true);
				try{
					ar.setUseAIC(true);
					armaTrain=ar.process(train);
					armaTest=ar.process(test);
					histoTrain=rl.process(train);
					histoTest=rl.process(test);
	
					
					fftTrain=fft.process(train);
					fftTest=fft.process(test);
					of.writeLine(fftTrain+"\n");
					
					fft.truncate(fftTrain,n/4);
					fft.truncate(fftTest,n/4);				
					fftAcc+=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),fftTrain,fftTest);
					histAcc+=ClassifierTools.singleTrainTestSplitAccuracy(new J48(),histoTrain,histoTest);
					armaAcc+=ClassifierTools.singleTrainTestSplitAccuracy(new IB1(),armaTrain,armaTest);
				}catch(Exception e){
					System.out.println("Error in process: e = "+e);
					e.printStackTrace();
					System.exit(0);
				}
//3. Measure classification accuracy on both 	
			}
			fftAcc/=reps;
			histAcc/=reps;
			armaAcc/=reps;
			System.out.println("FFT="+fftAcc+" Histo ="+histAcc+" ARMA ="+armaAcc);
			of.writeLine(histAcc+","+armaAcc);
		}	
	}

	
	
//Find an example 	
	public static void distanceMetricComparison(String file){
		
		
		
	}
	public static void AIC_Lengths(String fileName){
		//Generate a model
		int startN=100, endN=5000, increment=100;
		int[] cases ={200,1};
		OutFile of=new OutFile(fileName);
		of.writeString("n,ARMA_Length\n");
		System.out.print("n,ARMA_Length \n");
		//Model2: 0.76	0.995	
		double[][] paras={	{0.4881,0.6105,-0.2979},
			{0.8391}};
//		double[][] paras={		{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004},
//		{1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}		};	
		try{
			for(int n=startN;n<endN;n+=increment)
			{
		//1. Generate data
				of.writeString(n+",");
				System.out.print(n+",");
				Instances train =SimulateAR.generateARDataSet(paras,n,cases);
				ARMA ar = new ARMA();
				//2. Fit ARMA
				Instances arTrain=ar.process(train);
				//Find number of parameters
				for(int i=0;i<cases[0];i++)
				{
					Instance inst=arTrain.instance(i);
					int length=0;
					while(length<inst.numAttributes()&&inst.value(length)!=0)
						length++;
					of.writeString(length+",");
					System.out.print(length+",");
				}
				of.writeString("\n");
				System.out.print("\n");
					
			}
		}catch(Exception e){
			System.out.println("Error in Experiment 6, exit ");
			System.exit(0);
		}
		
		
	}
	
	public static void AIC_Effect(String fileName){
//Measure the effect of AIC length.
//1. Keep all
//2. Use AIC
//3. Fix to correct length (fix MAXLAG)		
		int startN=5000, endN=6000, increment=500;
		int[] cases ={100,100};
		int reps=2;
		OutFile of=new OutFile(fileName);
		of.writeString("n,ar_AIC,ar_All,ar_Correct\n");
		System.out.print("n,ar_All,ar_AIC,ar_Correct\n");
		double[][] paras=
		{	{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0,0.00,0.0004},
				{1.3532,0.4188,-1.2153,0.3091,0.0,-0.0,0.00,0.5004}};
			
			
			
//			exampleModels[0];
		DecimalFormat dc=new DecimalFormat("###.####");
		
		try{
			for(int f=0;f<1;f++){
//				paras=exampleModels[f];	
				System.out.println("\n\n MODEL NUMBER "+f+"\n\n");
				for(int n=startN;n<endN;n+=increment)
				{
					double a1=0,a2=0,a3=0;
					for(int i=0;i<reps;i++){
			//1. Generate data
						Instances train =SimulateAR.generateARDataSet(paras,n,cases);
						Instances test =SimulateAR.generateARDataSet(paras,n,cases);
	//Model 1, use all the parameters					
						ARMA ar = new ARMA();
						ar.setUseAIC(false);
						if(n<200)
							ar.setMaxLag(10);
						else
							ar.setMaxLag(20);
							
	//Model 2, use AIC fit					
						ARMA ar2 = new ARMA();
						ar2.setUseAIC(true);
	//Model 3, use correct number
						ARMA ar3 = new ARMA();
						ar3.setUseAIC(false);
						ar3.setMaxLag(8);
						//2. Fit ARMA
						Instances arTrain=ar.process(train);
						Instances ar2Train=ar2.process(train);
						Instances ar3Train=ar3.process(train);
						Instances arTest=ar.process(test);
						Instances ar2Test=ar2.process(test);
						Instances ar3Test=ar3.process(test);
						a1+=ClassifierTools.singleTrainTestSplitAccuracy(new J48(),arTrain,arTest);
						a2+=ClassifierTools.singleTrainTestSplitAccuracy(new J48(),ar2Train,ar2Test);
						a3+=ClassifierTools.singleTrainTestSplitAccuracy(new J48(),ar3Train,ar3Test);
					}
					of.writeLine(n+","+a1/reps+","+a2/reps+","+a3/reps);
					System.out.println(n+","+dc.format(a1/reps)+","+dc.format(a2/reps)+","+dc.format(a3/reps));
					
				}		
			}
		}catch(Exception e){
				System.out.println("Error in Experiment 6, exit ");
				System.exit(0);
		}
		
	}

//F1 Experiment: Generate sliding window histograms, recalculate histogram AND Spectrogram each time. Can be massively optimised by online calculation
	public static void f1HistogramsBasic(){
//Parameters: n =series length,w = window length
		int n=80000;	//629291;
		int w=8000; //8000 == 1 second
		int mrl=200;
		double[] data=new double[n];
		double[] window=new double[w];
		double[] oldWindow=new double[w];
		//Load sound data into array
		InFile f=new InFile("C:\\Research\\Data\\F1\\myF1.csv");
		OutFile of=new OutFile("C:\\Research\\Data\\F1\\basicdistancesF1.csv");
		OutFile of2=new OutFile("C:\\Research\\Data\\F1\\histoF1.csv");
		for(int i=0;i<n;i++)
			data[i]=f.readDouble();
		int[] histo,oldHisto;	
//Test histogram for first series
		System.arraycopy(data,0,oldWindow,0,w);
		RunLength rl=new RunLength();
		oldHisto=rl.processSingleSeries(oldWindow, mrl);
		
//Each step: 
		//1. Extract new series		
		//2. Get histogram
		//3. Measure distance between original series and histograms
		//4. Write to file
		
		for(int i=1;i<n-w;i++){
			window=new double[w];
			System.arraycopy(data,i,window,0,w);
//Histogram 		
			histo=rl.processSingleSeries(window, mrl);
//			for(int j=0;j<histo.length;j++)
//				of.writeString(histo[j]+",");
//			of.writeString("\n");
	//Compare current histo to old histo
			
			if(i%10000==0)
				System.out.println(" Finished step "+i);
			//Euclidean distance between raw data
			double d1=dist(window,oldWindow);
			//Euclidean distance between histograms
			double d2=dist(histo,oldHisto);
			of.writeLine(d1+","+d2);
			oldHisto=histo;
			oldWindow=window;			
		}	
	}
	public static double dist(double[] a, double[] b){
		double d=0;
		for(int i=0;i<a.length;i++)
			d+=(a[i]-b[i])*(a[i]-b[i]);
		return d;
	}
	public static double dist(int[] a, int[] b){
		double d=0;
		for(int i=0;i<a.length;i++)
			d+=(a[i]-b[i])*(a[i]-b[i]);
		return d;
	}
	
}	