/*
 * @author ajb
 * A class to evaluate shapelets for classification. The point of these experiment is to answer the question: what is the best way to deal with 
 * Shapelets, given its the optimal transform to use. Alternatives are
 * 
 * Shapelet tree, a la the original
 * Shapelet transform, re KDD paper
 */
package development;


import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import statistics.simulators.ShapeletModel;
import statistics.simulators.DataSimulator;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;
import weka.core.shapelet.QualityMeasures;
import weka.filters.*;
import weka.filters.timeseries.shapelet_transforms.*;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.CachedSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.OnlineSubSeqDistance;
/**
 *
 * @author ajb
 */
public class ShapeletClassification {
    
 	public static Classifier[] setSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<>();
		sc2.add(new kNN(1));
		names.add("NN");
               
		Classifier c=new SMO();
		PolyKernel kernel = new PolyKernel();
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
	
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
   
    
 
public static String path="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
    		public static String[] smallProblems={
			"Coffee", //28,28,286,2
			"Beef", //30,30,470,5
			"FaceFour",//24,88,350,4
            		"CBF",//30,900,128,3
			"fish",//175,175,463,7
			"Gun_Point",//50,150,150,2
			"OSULeaf", //200,242,427,6
			"synthetic_control", //300,300,60,6
			"Trace",//100,100,275,4
		};
  
   
// SIMULATED DATA    his method tests the base functionality of the class ShapeletGenerator   
    public static void generatorTest(){
/* This method tests the base functionality of the class ShapeletModel and FullShapeletTransform, somewhat reproducing the test harness experiments but put heree for tidiness and sanity 
*/
        System.out.println(" Test the ShapeletModel and its use with the DataGenerator class");
        ShapeletModel[] s=new ShapeletModel[2];
        int nosCases=400;
        int[] casesPerClass={nosCases/2,nosCases/2};
        int seriesLength=500;
        int shapeletLength=30;
//PARAMETER LIST:  numShapelets, seriesLength, shapeletLength, maxStart        
        double[] p1={1,seriesLength,shapeletLength};
        double[] p2={1,seriesLength,shapeletLength};
        
//Create two ShapeleModels with different base Shapelets        
        s[0]=new ShapeletModel(p1);        
        ShapeletModel.ShapeType st=s[0].getShapeType();
        s[1]=new ShapeletModel(p2);
        while(st==s[1].getShapeType())
            s[1]=new ShapeletModel(p2);
            
        System.out.println(" Shape 1= "+s[0]);
        System.out.println(" Shape 2= "+s[1]);
        
        DataSimulator ds=new DataSimulator(s);
        Instances train=ds.generateDataSet(seriesLength,casesPerClass);
        Instances test=ds.generateDataSet(seriesLength,casesPerClass);
        Classifier base=new J48();
        double a=ClassifierTools.singleTrainTestSplitAccuracy(base, train, test);
        System.out.println(" Accuracy ="+a);
    }

    public static void earlyAbandonDebug(){
//For some reason, synthetic control and lightning 7 take forever with EA. 
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\SyntheticControl\\SyntheticControl_TRAIN");
        FullShapeletTransform s1=new FullShapeletTransform();
        s1.setSubSeqDistance(new OnlineSubSeqDistance());
        FullShapeletTransform s2=new FullShapeletTransform();
        s2.setSubSeqDistance(new OnlineSubSeqDistance());
//        ShapeletExamples.initializeShapelet(s1, train);
        //        {20, 56},   // SyntheticControl
        s1.setShapeletMinAndMax(20, 56);
        s1.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        s1.setNumberOfShapelets((train.numAttributes()-1)/2);        
        s1.supressOutput();
        s1.setCandidatePruning(false);
        s2.setShapeletMinAndMax(20, 56);
        s2.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        s2.setNumberOfShapelets((train.numAttributes()-1)/2);        
        s2.supressOutput();
        s2.setCandidatePruning(true);
        long t1,t2;
        double time1,time2;
        DecimalFormat df =new DecimalFormat("###.####");
        try {
                t1=System.currentTimeMillis();
                s1.process(train);
                t2=System.currentTimeMillis();
                time1=((t2-t1)/1000.0);
                s1.setCandidatePruning(true);
                t1=System.currentTimeMillis();
                s2.process(train);
                t2=System.currentTimeMillis();
                time2=((t2-t1)/1000.0);
                System.out.println(" ********* QUALITY MEASURE ="+s1.getQualityMeasure()+"  **********");
                System.out.println(" NO ABANDON \t\t ABANDON\t\t ABANDON/(NO ABANDON)%\t\t SPEED UP ");
                System.out.println(df.format(time1)+"\t\t\t"+df.format(time2)+"\t\t\t"+(int)(100.0*(time2/time1))+"%"+"\t\t\t"+df.format(time1/time2));
       } catch (Exception ex) {
            System.out.println("Error performing the shapelet transform"+ex);
            ex.printStackTrace();
            System.exit(0);
        }                   

        
    }
		public static String[] nonFastShapeletProblems={	
                        "ArrowHead",                        
			"ARSim", // 2000,2000,500,2
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
                        "Car",
			"CBF", // 30,900,128,3
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
                        "DistalPhalanxOutlineCorrect",
                        "DistalPhalanxOutlineAgeGroup",
//                        "DistalPhalanxOIntensityAgeGroup",
                        "DistalPhalanxTW",
			"Earthquakes", // 322,139,512,2
			"ElectricDevices", // 8953,7745,96,7
			"FaceAll", // 560,1690,131,14
			"fiftywords", // 450,455,270,50
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"HandOutlines", // 1000,370,512,2
			"Herring", // 64,64,512,2
			"Haptics", // 155,308,1092,5
			"InlineSkate", // 100,550,1882,7
                        "MiddlePhalanxOutlineCorrect",
                        "MiddlePhalanxOutlineAgeGroup",
                        "MiddlePhalanxTW",
                        "NonInvasiveFatalECG_Thorax1",
                        "NonInvasiveFatalECG_Thorax2",
                        "PhalangesOutlinesCorrect",
//                      "PassGraphs,  
                        "Plane",
                        "ProximalPhalanxOutlineCorrect",
                        "ProximalPhalanxOutlineAgeGroup",
                        "ProximalPhalanxTW",
			"ShapesAll", // 600,600,512,60
			"StarLightCurves", // 1000,8236,1024,3
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
                        "ToeSegmentation1",
                        "ToeSegmentation2",
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
                        "WordSynonyms"
                };
/* Problem sets used in @article{rakthanmanon2013fast,
  title={Fast Shapelets: A Scalable Algorithm for Discovering Time Series Shapelets},
  author={Rakthanmanon, T. and Keogh, E.},
  journal={Proceedings of the 13th {SIAM} International Conference on Data Mining},
  year={2013}
}
All included except Cricket. There are three criket problems and they are not 
* alligned, the class values in the test set dont match

*/
		public static String[] fastShapeletProblems={	
			"ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SonyAIBORobotSurface", // 20,601,70,2
			"Beef", // 30,30,470,5
			"GunPoint", // 50,150,150,2
			"TwoLeadECG", // 23,1139,82,2
                        "Adiac", // 390,391,176,37
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"Coffee", // 28,28,286,2
			"DiatomSizeReduction", // 16,306,345,4
			"ECGFiveDays", // 23,861,136,2
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fish", // 175,175,463,7
			"Lighting2", // 60,61,637,2
			"Lighting7", // 70,73,319,7
			"FaceAll", // 560,1690,131,14
			"MALLAT", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"wafer", // 1000,6164,152,2
                        "yoga",
                        "FaceAll",
                        "TwoPatterns",
        		"CinC_ECG_torso" // 40,1380,1639,4
                };

 		public static String[] missingFastShapeletProblems={	
 //			"fish", // 175,175,463,7
 //       		"CinC_ECG_torso", // 40,1380,1639,4
//			"MALLAT", // 55,2345,1024,8
//			"OSULeaf", // 200,242,427,6
                        "yoga"
                };
    public static void ucrDataSets(String resultsPath, boolean saveTransforms){
             DecimalFormat df = new DecimalFormat("###.###");
            OutFile of = new OutFile(resultsPath);
            System.out.println("************** SHAPELET TRANSFORM ON UCR*******************");
            ArrayList<String> names=new ArrayList<>();
            Classifier[] c= setSingleClassifiers(names);
            for(String s:names){
                of.writeString(s+",");
                System.out.print(s+"\t");
            }
                of.writeString("\n");
                System.out.print("\n");
                for(int i=0;i<fastShapeletProblems.length;i++)
                {
                    FullShapeletTransform s=null;
                     Instances test=null;
                     Instances train=null;
                    try{
                           test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fastShapeletProblems[i]+"\\"+fastShapeletProblems[i]+"_TEST");
                            train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fastShapeletProblems[i]+"\\"+fastShapeletProblems[i]+"_TRAIN");			
                            OutFile o2=null,o3=null;
                            if(saveTransforms){
                                File f = new File("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]);
                                if(!f.isDirectory())//Test whether directory exists
                                    f.mkdir();
                                o2=new OutFile("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]+"\\Shapelet"+fastShapeletProblems[i]+"_TRAIN.arff");
                                o3=new OutFile("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]+"\\Shapelet"+fastShapeletProblems[i]+"_TEST.arff");
                            }
                            System.gc();
                            ShapeletTransformFactory f =new ShapeletTransformFactory();

                            s=f.createTransform(train);
                            if(saveTransforms){
                                s.setLogOutputFile("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]+"\\Shapelet"+fastShapeletProblems[i]+"logFile.txt");
                            }
                            else
                                s.turnOffLog();
                            System.out.print("Transforming train "+fastShapeletProblems[i]+" .... ");

                           train=s.process(train);
                           System.gc();
                            if(saveTransforms){
    //Need to do this instance by instance to save memory
                                Instances header=new Instances(train,0);
                                o2.writeLine(header.toString());
                                o3.writeLine(header.toString());
                                for(int j=0;j<train.numInstances();j++)
                                    o2.writeLine(train.instance(j).toString());
                                System.out.print("Transforming test "+fastShapeletProblems[i]+" .... ");
                                for(int j=0;j<test.numInstances();j++){
                                    Instances testTemp=new Instances(test,0);
                                    testTemp.add(test.instance(i));    
                                    testTemp=s.process(testTemp);
                                    o3.writeLine(testTemp.toString());
                                    System.gc();
                                }
                            }
                            else{
                                test=s.process(test);
                            }

    //Save results to file
    //Train Classifiers
                            System.out.print(" Classifying .... ");
                            c= setSingleClassifiers(names);
                            of.writeString(fastShapeletProblems[i]+",");
                            for(int j=0;j<c.length;j++){
                                c[j].buildClassifier(train);
                                double a=utilities.ClassifierTools.accuracy(test,c[j]);
                                System.out.print(a+"\t");
                                of.writeString(a+",");
                            }                                    
                                System.out.print("\n");
                                of.writeString("\n");
                    }catch(Exception e){
                            System.out.println(" Error in accuracy ="+e);
                            e.printStackTrace();
                            System.exit(0);
                    } 
               }     
                                
             }
/* Test that  the round robin version correctly retains the index of the
    originaltraining data
    */    
    public static void testRoundRobin(){
        String problem="ItalyPowerDemand";
        Instances train=ClassifierTools.loadData(path+problem+"\\"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(path+problem+"\\"+problem+"_TEST");
        FullShapeletTransform st=new FullShapeletTransform();
        st.setUseRoundRobin(false);
        st.supressOutput();
        FullShapeletTransform st2=new FullShapeletTransform();
        st2.setUseRoundRobin(true);
        st2.supressOutput();
        FullShapeletTransform st3=new FullShapeletTransform();
        st3.setSubSeqDistance(new OnlineSubSeqDistance());
         try {
             Instances train2=st.process(train);
             Instances train3=st2.process(train);
             Instances train4=st3.process(train);
             for(int i=0;i<train2.numInstances();i++)
                     System.out.println(train2.instance(i).classValue()+","+train3.instance(i).classValue()+","+train4.instance(i).classValue());
             for(int i=0;i<train2.numInstances();i++)
                    if(train2.instance(i).classValue()!=train3.instance(i).classValue()||train3.instance(i).classValue()!=train4.instance(i).classValue()){
                     System.out.println(" MISMATCH");
                     System.out.println(i+","+train2.instance(i).classValue()+","+train3.instance(i).classValue());
                    }
             OutFile of=new OutFile("NoRRTest.arff");
             of.writeLine(st.toString());
             of=new OutFile("RRTest.arff");
             of.writeLine(st2.toString());
             of=new OutFile("ROnlineNormTest.arff");
             of.writeLine(st3.toString());
    //        Instances 
         } catch (Exception ex) {
             Logger.getLogger(ShapeletClassification.class.getName()).log(Level.SEVERE, null, ex);
         }
    }
    
 
   public static void testMemoryCache(){
        String problem="ItalyPowerDemand";
        Instances train=ClassifierTools.loadData(path+problem+"\\"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(path+problem+"\\"+problem+"_TEST");
        FullShapeletTransform st=new FullShapeletTransform();
        st.setSubSeqDistance(new OnlineSubSeqDistance());
        st.setUseRoundRobin(false);
        st.supressOutput();
        FullShapeletTransform st2=new FullShapeletTransform();
        st2.setSubSeqDistance(new OnlineSubSeqDistance());
        st2.setUseRoundRobin(true);
        st2.supressOutput();
        FullShapeletTransform st3=new FullShapeletTransform();
        st3.setSubSeqDistance(new CachedSubSeqDistance());
        st3.setUseRoundRobin(true);
         try {
             Instances train2=st.process(train);
             Instances train3=st2.process(train);
             Instances train4=st3.process(train);
             for(int i=0;i<train2.numInstances();i++)
                     System.out.println(train2.instance(i).classValue()+","+train3.instance(i).classValue()+","+train4.instance(i).classValue());
             for(int i=0;i<train2.numInstances();i++)
                    if(train2.instance(i).classValue()!=train3.instance(i).classValue()||train3.instance(i).classValue()!=train4.instance(i).classValue()){
                     System.out.println(" MISMATCH");
                     System.out.println(i+","+train2.instance(i).classValue()+","+train3.instance(i).classValue());
                    }
             OutFile of=new OutFile("NoRRTest.arff");
             of.writeLine(st.toString());
             of=new OutFile("RRTest.arff");
             of.writeLine(st2.toString());
             of=new OutFile("RDistCacheTest.arff");
             of.writeLine(st3.toString());
             
             Instances test2=st.process(test);
             Instances test3=st2.process(test);
             Instances test4=st3.process(test);
             
             Classifier c = new kNN();
             double a2=ClassifierTools.singleTrainTestSplitAccuracy(c, train2, test2);
             double a3=ClassifierTools.singleTrainTestSplitAccuracy(c, train3, test3);
             double a4=ClassifierTools.singleTrainTestSplitAccuracy(c, train4, test4);
                System.out.println(" a2= "+a2+" a3 = "+a3+" a4 = "+a4);
             
    //        Instances 
         } catch (Exception ex) {
             Logger.getLogger(ShapeletClassification.class.getName()).log(Level.SEVERE, null, ex);
         }
    }
    
     
    
    public static void generateTransforms(String resultsPath){
             DecimalFormat df = new DecimalFormat("###.###");
            OutFile of = new OutFile(resultsPath);
            System.out.println("************** SHAPELET TRANSFORM ON UCR*******************");
            for(int i=0;i<fastShapeletProblems.length;i++)
            {
                FullShapeletTransform s=null;
                Instances test=null;
                Instances train=null;

                test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fastShapeletProblems[i]+"\\"+fastShapeletProblems[i]+"_TEST");
                train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+fastShapeletProblems[i]+"\\"+fastShapeletProblems[i]+"_TRAIN");			
                OutFile o2=null,o3=null;
                File f = new File("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]);
                if(!f.isDirectory())//Test whether directory exists
                    f.mkdir();
                o2=new OutFile("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]+"\\Shapelet"+fastShapeletProblems[i]+"_TRAIN.arff");
                o3=new OutFile("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]+"\\Shapelet"+fastShapeletProblems[i]+"_TEST.arff");
                System.gc();
                ShapeletTransformFactory fact =new ShapeletTransformFactory();
                s=fact.createTransform(train);
                s.setLogOutputFile("C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\Shapelet"+fastShapeletProblems[i]+"\\Shapelet"+fastShapeletProblems[i]+"logFile.txt");
                System.out.print("Transforming train "+fastShapeletProblems[i]+" .... ");
                Instances header;
                try{
                    train=s.process(train);
                    System.gc();
                    header=new Instances(train,0);
                    o2.writeLine(header.toString());
                for(int j=0;j<train.numInstances();j++)
                    o2.writeLine((train.instance(j)).toString());
                }catch(Exception e){
                    System.out.println("Exception "+e+" creating train transform for "+fastShapeletProblems[i]);
                    e.printStackTrace();
                    System.exit(0);
                }
                header=new Instances(train,0);
                o3.writeLine(header.toString());
//Need to do this instance by instance to save memory
                System.out.print("Transforming test "+fastShapeletProblems[i]+" .... ");
                for(int j=0;j<test.numInstances();j++){
                    try{
                        Instances testTemp=new Instances(test,0);
                        testTemp.add(test.instance(j));    
                        testTemp=s.process(testTemp);
                        o3.writeLine((testTemp.instance(0)).toString());
                        System.gc();
                    }catch(Exception e){
                        System.out.println("Exception "+e+" creating test transform data number "+j+" for "+fastShapeletProblems[i]);
                        e.printStackTrace();
                        System.exit(0);
                    }
                }
          }
    }
    public static void transformOnCluster(String fileName){
        String directoryName="TSC_Problems";
        String resultName="ShapeletTransformedTSC_Problems";
        DecimalFormat df = new DecimalFormat("###.###");
        System.out.println("************** SHAPELET TRANSFORM ON UCR*******************");
        FullShapeletTransform s=null;
        Instances test=null;
        Instances train=null;
        test=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/"+fileName+"/"+fileName+"_TEST");
        train=utilities.ClassifierTools.loadData(DataSets.clusterPath+directoryName+"/"+fileName+"/"+fileName+"_TRAIN");			
        OutFile o2=null,o3=null;
        o2=new OutFile(DataSets.clusterPath+resultName+"/Shapelet"+fileName+"/Shapelet"+fileName+"_TRAIN.arff");
        o3=new OutFile(DataSets.clusterPath+resultName+"/Shapelet"+fileName+"/Shapelet"+fileName+"_TEST.arff");
        ShapeletTransformFactory fact =new ShapeletTransformFactory();
        s=fact.createTransform(train);
        s.setLogOutputFile(DataSets.clusterPath+resultName+"/Shapelet"+fileName+"/Shapelet"+fileName+"logFile.txt");
        System.out.print("Transforming train "+fileName+" .... ");
        Instances header;
        try{
            train=s.process(train);
            header=new Instances(train,0);
            o2.writeLine(header.toString());
            for(int j=0;j<train.numInstances();j++)
                o2.writeLine(train.instance(j).toString());
        }catch(Exception e){
            System.err.println("Exception "+e+" creating train transform for "+fileName);
            e.printStackTrace();
            System.exit(0);
        }
        header=new Instances(train,0);
        o3.writeLine(header.toString());
//Need to do this instance by instance to save memory
        System.out.print("Transforming test "+fileName+" .... ");
        for(int j=0;j<test.numInstances();j++){
            try{
                Instances testTemp=new Instances(test,0);
                testTemp.add(test.instance(j));    
                testTemp=s.process(testTemp);
                o3.writeLine(testTemp.toString());
                System.gc();
            }catch(Exception e){
                System.err.println("Exception "+e+" creating test transform data number "+j+" for "+fileName);
                e.printStackTrace();
                System.exit(0);
            }
        }
    }
    public static void clusterTest(){
           for(int i=0;i<nonFastShapeletProblems.length;i++)
            {
                File f = new File("C:\\Users\\ajb\\Dropbox\\Temp\\Shapelet"+nonFastShapeletProblems[i]);
                if(!f.isDirectory())//Test whether directory exists
                    f.mkdir();    
            }
    }
    
    public static void shapeletTestSetAccuracy(String outfile){
        OutFile of= new OutFile(outfile);
        String path="C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\";
        String qualifier="Shapelet";
        String[] fileNames=fastShapeletProblems;
        ArrayList<String> names= new ArrayList<>();
        of.writeString(",");
        for(String s:names)
            of.writeString(s+",");
        of.writeString("\n");
        try{
            for(int i=29;i<fileNames.length;i++){
                Classifier[] c=setSingleClassifiers(names);
                System.out.println(" Problem file ="+fileNames[i]);
                    of.writeString(qualifier+fastShapeletProblems[i]+",");
                    Instances test=utilities.ClassifierTools.loadData(path+qualifier+fastShapeletProblems[i]+"\\"+qualifier+fastShapeletProblems[i]+"_TEST2");
                    Instances train=utilities.ClassifierTools.loadData(path+qualifier+fastShapeletProblems[i]+"\\"+qualifier+fastShapeletProblems[i]+"_TRAIN");			
                    for(int j=0;j<c.length;j++){
                        c[j].buildClassifier(train);
                        double acc=ClassifierTools.accuracy(test, c[j]);
                        of.writeString(acc+",");
                    }
                    of.writeString("\n");
            }
        }
        catch(Exception e){
                System.err.println("Exception "+e+" loading shapoelets");
                e.printStackTrace();
                System.exit(0);
        }
        
    }   
    public static void parseShapelets(){
        String path="C:\\Users\\ajb\\Dropbox\\Shapelet Transformed TSC Problems\\";
        String qualifier="Shapelet";
        String[] fileNames=fastShapeletProblems;    //TimeSeriesClassification.fileNames;
        for(int i=29;i<fileNames.length;i++){
            InFile f=new InFile(path+qualifier+fileNames[i]+"\\"+qualifier+fileNames[i]+"_TEST.arff");
            OutFile of=new OutFile(path+qualifier+fileNames[i]+"\\"+qualifier+fileNames[i]+"_TEST2.arff");
            String str="";
            boolean header=true;
            int j=0;
            str=f.readLine();
            while(str!=null && header){
                of.writeLine(str);
                if(str.equals("@data"))
                    header=false;
                else
                    str=f.readLine();
                j++;
            }
            str=f.readLine();
            while(str!=null){
                of.writeLine(str);  //Valid line read
//Read up to the next @data
                str=f.readLine();
                while(str!=null&&!str.equals("@data")){
                        str=f.readLine();
                    }
                    str=f.readLine();
                }
         }
        
        
    }
    
    public static void beefTest() throws Exception {
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\Beef\\Beef_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\Beef\\Beef_TEST");
        NormalizeCase nc = new NormalizeCase();
        nc.setNormType(NormalizeCase.NormType.STD_NORMAL);
        Instances normTrain=nc.process(train);
        Instances normTest=nc.process(test);
        
        FullShapeletTransform st =new FullShapeletTransform(); 
        st.setNumberOfShapelets((normTrain.numAttributes()-1)/2);
        st.setShapeletMinAndMax(8, 30);
        Instances shapeTrain=st.process(train);
        Instances shapeTest=st.process(test);
        
        st =new FullShapeletTransform(); 
        st.setNumberOfShapelets((normTrain.numAttributes()-1)/2);
        st.setShapeletMinAndMax(8, 30);
        
        Instances shapeNormTrain=st.process(normTrain);
        Instances shapeNormTest=st.process(normTest);
        
        ArrayList<String> names= new ArrayList<>();
        Classifier[] c=ClassifierTools.setSingleClassifiers(names);
        double[] acc1=new double[c.length];
        double[] acc2=new double[c.length];
        
        for(int i=0;i<c.length;i++){
            c[i].buildClassifier(shapeTrain);
            acc1[i]=ClassifierTools.accuracy(shapeTest, c[i]);
            c[i].buildClassifier(shapeNormTrain);
            acc2[i]=ClassifierTools.accuracy(shapeNormTest, c[i]);
            System.out.println(" Classifier "+names.get(i)+" Std = "+acc1[i]+" normed ="+acc2[i]);
        }
        
                
    } 
            public static void main(String[] args){
                Instances sony = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\SonyAIBORobotSurfaceII\\SonyAIBORobotSurfaceII_TRAIN");
                OutFile test = new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\SonyAIBORobotSurfaceII\\test.arff");
             try {
                 generateShapeletFile(sony,test);
             } catch (Exception ex) {
                 Logger.getLogger(ShapeletClassification.class.getName()).log(Level.SEVERE, null, ex);
             }
                System.exit(0);
              
                testMemoryCache();                
                
                try{
                    beefTest();
                }
                catch(Exception e){
                    System.out.println("Exception ="+e);
                }
                System.exit(0);
//                parseShapelets();
                shapeletTestSetAccuracy("C:\\Users\\ajb\\Dropbox\\Results\\ShapeletDomain\\FastShapeletProblems.csv");
                 int index=Integer.parseInt(args[0])-1;
            System.out.println("Input ="+index);
           transformOnCluster(missingFastShapeletProblems[index]);
 
  //              generatorTest();
 //               earlyAbandonDebug();
//                    ucrDataSets("C:\\Users\\ajb\\Dropbox\\Results\\Shapelets\\FastShapeletsRecreation.csv",true);
//                    generateTransforms("C:\\Users\\ajb\\Dropbox\\Results\\Shapelets\\FastShapeletsRecreation.csv");
            }
public static void generateShapeletFile(Instances train, OutFile dest) throws Exception{
            //Shapelet
        FullShapeletTransform s2=new FullShapeletTransform();
        s2.setSubSeqDistance(new OnlineSubSeqDistance());
        s2.setNumberOfShapelets(train.numInstances()*10);
        s2.setShapeletMinAndMax(3, train.numAttributes()-1);
        s2.setDebug(false);
        s2.supressOutput();
        s2.turnOffLog();  
        s2.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.F_STAT);
        Instances shapeletTrain=s2.process(train);
        dest.writeLine(shapeletTrain+"");
}


}