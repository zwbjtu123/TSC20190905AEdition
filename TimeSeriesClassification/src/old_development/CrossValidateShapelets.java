/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package old_development;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.HeterogeneousEnsemble;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformDistCaching;

/**
 *
 * @author ajb
 */
public class CrossValidateShapelets extends Thread {
    Instances train;
    Instances test;
    int fold;
    String path;
    public static String fileName;
    public static boolean useCluster=true;
    //UCR ONES FIRST 
    static int[] missing={45,46,61,71,72,73,74,20,26,27,57};
    static int[] incomplete={45,46,61,71,72,73,74,20,26,27,57};
    
    
    
    public CrossValidateShapelets(Instances tr, Instances te, int f,String path){
        train=tr;
        test=te;
        fold=f;
        this.path=path;
    }
    public static void formCV(){
//Delete any existing shapelet files for the incomplete
        for(int i=0;i<incomplete.length;i++){
             File f = new File("/gpfs/sys/ajb/TSC Problems/"+DataSets.fileNames[incomplete[i]]+"/ShapeletCV/");
//Delete everything there             
            if(f.exists()){
                try{
                    delete(f);
                }catch(IOException e){
                    System.err.println(" Unable to delete directory ShapeletCV/  Continuing  ");
                }
            }
//Recreate the directory             
            if(!f.exists()){
                f.mkdir();            
            }
        }
        for(int i=0;i<missing.length;i++){
            String clusterPath = "/gpfs/sys/ajb/TSC Problems/"+DataSets.fileNames[missing[i]]+"/";
            String dropboxPath= "C:/Users/ajb/Dropbox/TSC Problems/"+DataSets.fileNames[missing[i]]+"/";
//            String path=dropboxPath;
            String path=clusterPath;            
            Instances train = ClassifierTools.loadData(path+DataSets.fileNames[missing[i]]+"_TRAIN");
            System.out.println("Processing : "+DataSets.fileNames[missing[i]]);
            NormalizeCase nc = new NormalizeCase();
            try{
                train=nc.process(train);
            }catch(Exception e){
                System.out.println(" Unable to normalise for some unknown reason "+e+"  but continuing...");
            }
//Randomize the data. Need to save the mapping somewhere.
            int[] positions=new int[train.numInstances()];
           train=randomise(train,positions);
            OutFile of = new OutFile(path+"ShapeletCV/InstancePositions.csv");
            for(int j=0;j<positions.length;j++)
                of.writeLine(positions[j]+",");           
            of = new OutFile(path+"InstancePositions.csv");
            for(int j=0;j<positions.length;j++)
                of.writeLine(positions[j]+",");           
            
//Split into time domain folds
            int folds=10;
            Instances[] trainFolds=new Instances[folds];
            Instances[] testFolds=new Instances[folds];
            splitTrainData(train,trainFolds,testFolds,folds);
//Save folds to file
            for(int j=1;j<=folds;j++){
                OutFile of1 =  new OutFile(path+DataSets.fileNames[missing[i]]+"_TRAIN"+(j)+".arff");
              OutFile of2 =  new OutFile(path+DataSets.fileNames[missing[i]]+"_TEST"+(j)+".arff");
              of1.writeLine(trainFolds[j-1].toString());
              of2.writeLine(testFolds[j-1].toString());
            }
            
        }
    }
    
    
    public void run(){
//Perform cached on online 
        FullShapeletTransform st=new ShapeletTransformDistCaching();
        st.useCandidatePruning(10);
//        if(train.numInstances()>=500 || train.numAttributes()>500)
//            st = new ShapeletTransform();
        st.supressOutput();
        st.setNumberOfShapelets(Math.max(train.numAttributes(), train.numInstances()));
        try {
            Instances sTrain=st.process(train);
            Instances sTest=st.process(test);
            OutFile of1 =  new OutFile(path+fileName+"_TRAIN"+(fold+1)+".arff");
            OutFile of2 =  new OutFile(path+fileName+"_TEST"+(fold+1)+".arff");
            of1.writeLine(sTrain.toString());
            of2.writeLine(sTest.toString());
            
        } catch (Exception ex) {
            Logger.getLogger(CrossValidateShapelets.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    public static void splitTrainData(Instances train,Instances[] trainFolds,Instances[] testFolds,int folds){
        int size=train.numInstances();
        int foldSize=size/folds;
        int[] foldCV = new int[folds];
        for(int i=0;i<foldCV.length;i++)
            foldCV[i]=foldSize;
        if(size%folds!=0)   //Adjust the last fold size accordingly
            foldCV[folds-1]=size -foldSize*(folds-1); 
        int diff=foldCV[folds-1]-foldSize;
        int c=0;
        while(diff>0){  //Reassign elements to other folds
           
            foldCV[c%(folds-1)]++;
            foldCV[folds-1]--;
            diff=foldCV[folds-1]-foldCV[c%(folds-1)];
            c++;
        }
        Instances copy = new Instances(train);
        int start=0;
        for(int i=0;i<folds;i++){
            trainFolds[i]= new Instances(copy,0);
            testFolds[i]= new Instances(copy,0);
            for(int j=0;j<train.numInstances();j++){
                if(j<start || j>=start+foldCV[i])
                    trainFolds[i].add(train.instance(j));
                else
                    testFolds[i].add(train.instance(j));
            }
            start+=foldCV[i];
        }
    }
    public static Instances randomise(Instances train, int[] pos){
//Generate a random permutation into pos
        Random r = new Random();
        for(int i=0;i<pos.length;i++)
            pos[i]=i;
        for(int i=0;i<pos.length;i++){
            int p1=r.nextInt(pos.length);
            int p2=r.nextInt(pos.length);
            int temp=pos[p1];
            pos[p1]=pos[p2];
            pos[p2]=temp;
        }
        Instances newD=new Instances(train,0);
        for(int i=0;i<pos.length;i++)
            newD.add(train.instance(pos[i]));
        return newD;
    }

    
    public static void singleRunThreaded(String file){
//        String file ="ItalyPowerDemand";
        
        String clusterPath = "/gpfs/sys/ajb/TSC Problems/"+file+"/";
        String desktopPath="C:/Users/ajb/Dropbox/TSC Problems/"+file+"/";
        String path=desktopPath;
        if(useCluster)
            path=clusterPath;
        
        String filePath=path+"ShapeletCV/";

        int count=0;
//Create directory if it isn't there already          
        File dir = new File(filePath);
        if(!dir.exists()){
            dir.mkdir();
        }
        else{ //Comment out to allow overwriting 
            boolean present=true;
            for(int i=1;i<=10&& present;i++){
                File cv=new File(filePath+file+"_TRAIN"+i+".arff");
                File cv2=new File(filePath+file+"_TEST"+i+".arff");
                if(cv.exists()&&cv2.exists()) {
                    //CV files already there
                    count++;
                }
                else
                    present=false;
            }
            if(count==10)//Exit now
                return;
        }
        
      
        

        CrossValidateShapelets.fileName=file;
        Instances train = ClassifierTools.loadData(path+file+"_TRAIN");
        NormalizeCase nc = new NormalizeCase();
        try{
            train=nc.process(train);
        }catch(Exception e){
            System.out.println(" Unable to normalise for some unknown reason "+e+"  but continuing...");
        }
//Randomize the data. Need to save the mapping somewhere.
        int[] positions=new int[train.numInstances()];
       train=randomise(train,positions);
        OutFile of = new OutFile(filePath+"InstancePositions.csv");
        for(int i=0;i<positions.length;i++)
            of.writeLine(positions[i]+",");
        

//Split data into folds
        
        int folds=10;
        Instances[] trainFolds=new Instances[folds];
        Instances[] testFolds=new Instances[folds];
        splitTrainData(train,trainFolds,testFolds,folds);

        CrossValidateShapelets[] ct= new CrossValidateShapelets[folds];
        for(int i=0;i<folds;i++){
            ct[i]=new CrossValidateShapelets(trainFolds[i],testFolds[i],i,filePath);
        }
        for(int i=0;i<folds;i++){ //Only start the threads where file is not their
            ct[i].start();
        }
        try {
        for(int i=0;i<folds;i++)
                ct[i].join();
        } catch (InterruptedException ex) {
            Logger.getLogger(CrossValidateShapelets.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    public static int countFiles(String file){
        String path= "/gpfs/sys/ajb/TSC Problems/"+file+"/";
        String filePath=path+"ShapeletCV/";
//See if it has already done the job. If so, dont bother!
            boolean b=false;
            int count=0;
            for(int j=1;j<=10;j++){
                File cv=new File(filePath+file+"_TRAIN"+j+".arff");
                if(cv.exists())     //CV files already there
                    count++;
            }
            return count;
            
    }
    
    public static void checkTransforms(){
        int totalCount=0;
        for(int i=0;i<DataSets.fileNames.length;i++){ 
            String path= "/gpfs/sys/ajb/TSC Problems/"+DataSets.fileNames[i]+"/";
            String filePath=path+"ShapeletCV/";
//See if it has already done the job. If so, dont bother!
                boolean b=false;
                int count=0;
                for(int j=1;j<=10;j++){
                    File cv=new File(filePath+DataSets.fileNames[i]+"_TRAIN"+j+".arff");
                    if(cv.exists())     //CV files already there
                        count++;
                }
                if(count==10)
                    totalCount++;
                else
                    System.out.println("PROBLEM"+DataSets.fileNames[i]+" IN POSITION "+i+" ONLY "+count+" CV FILES COMPLETED");
        } 
        System.out.println("TOTAL COMPLETED = "+totalCount);
        
    }
    public static int[][] classifyFold(String file, int fold){
        
        String clusterPath = "/gpfs/sys/ajb/TSC Problems/"+file+"/";
        String desktopPath="C:/Users/ajb/Dropbox/TSC Problems/"+file+"/";
        String path=desktopPath;
        if(useCluster)
            path=clusterPath;
        
        String filePath=path+"ShapeletCV/";
        
        //Check training and test files exist, terminate if not
        File tr=new File(filePath+file+"_TRAIN"+fold+".arff");
        File ts=new File(filePath+file+"_TEST"+fold+".arff");
        if(!tr.exists() || !ts.exists()){
            System.err.println(" ERROR CLASSIFYING "+file+" fold "+fold+" file does not exist");
            return null;
        }
        //Check whether predictions exist, terminate if not.
        File r=new File(filePath+file+"Predictions"+fold+".csv");
        if(r.exists()){
            System.err.println(file+" fold "+fold+" Classificastion already done");
            return null;
        }
        
        Instances train = ClassifierTools.loadData(filePath+file+"_TRAIN"+fold);
        Instances test = ClassifierTools.loadData(filePath+file+"_TEST"+fold);
        ArrayList<String> names= new ArrayList<>();
        ArrayList<Classifier> c=setSingleClassifiers(names);
        HeterogeneousEnsemble hc = new HeterogeneousEnsemble(c);
        hc.useCVWeighting(true);
        
        int[][] preds=new int[2][test.numInstances()];
        try {
            hc.buildClassifier(train);
            for(int i=0;i<test.numInstances();i++){
                preds[0][i]=(int)test.instance(i).classValue();
                preds[1][i]=(int)hc.classifyInstance(test.instance(i));
            }
        } catch (Exception ex) {
            Logger.getLogger(CrossValidateShapelets.class.getName()).log(Level.SEVERE, null, ex);
        }
//Save results to the appropriate file
        double[] cvAccs=hc.getWeights();
        OutFile results=new OutFile(filePath+file+"Predictions"+fold+".csv");
        for(int i=0;i<cvAccs.length;i++)
            results.writeString(cvAccs[i]+",");
        results.writeString("\n Actual,Predicted\n");
        int correct=0;
        for(int i=0;i<preds[0].length;i++){
            results.writeString(preds[0][i]+","+preds[1][i]+"\n");
            if(preds[0][i]==preds[1][i])
                correct++;
        }
        System.out.println(" Fold ="+fold+" correct ="+correct+" acc = "+((double)correct)/preds[0].length);
        return preds;
        
        
    }
    
    public static void combineandInvertFolds(){
        OutFile all=new OutFile("/gpfs/sys/ajb/shapeletCV/TrainCV.csv");
//        OutFile all=new OutFile("C:/Users/ajb/Dropbox/TSC Problems/TrainCV.csv");
        fileLoop:for(int i=0;i<DataSets.fileNames.length;i++)
        {
            all.writeString("\n"+DataSets.fileNames[i]);
//Check predictions exist, if not, ignore
            String path="/gpfs/sys/ajb/TSC Problems/";
//            String path="C:/Users/ajb/Dropbox/TSC Problems/";
            int count=0;
            for(int j=1;j<=10;j++){
                File f=new File(path+DataSets.fileNames[i]+"/ShapeletCV/"+DataSets.fileNames[i]+"Predictions"+j+".csv");
                if(f.exists())
                   count++; 
            }
            if(count<10){    //Skip this problem
                System.out.println(" Not enough Prediction files for problem "+DataSets.fileNames[i]+" num ="+count);
                continue fileLoop;
            }
//Check if combined file exists. If it does, do nothing.
 //           File f=new File("/gpfs/sys/ajb/shapeletCV/"+DataSets.fileNames[i]+"Preds.csv");
//            if(f.exists())//Skip this problem
//                continue fileLoop;
//Concatinate into a single file
            String str= "/gpfs/sys/ajb/shapeletCV/";
//            String str= "/gpfs/sys/ajb/TSC Problems/";
            OutFile of=new OutFile(str+DataSets.fileNames[i]+"Preds.csv");
            of.writeLine("actual,predicted");
            OutFile of2=new OutFile(str+DataSets.fileNames[i]+"CV_Accs.csv");
            ArrayList<int[]> preds=new ArrayList<>();
            int lines;
            InFile inF;
            for(int j=1;j<=10;j++){
                inF=new InFile(path+DataSets.fileNames[i]+"/ShapeletCV/"+DataSets.fileNames[i]+"Predictions"+j+".csv");
                lines =inF.countLines()-2;
                System.out.println(" Number of lines ="+lines);
                inF=new InFile(path+DataSets.fileNames[i]+"/ShapeletCV/"+DataSets.fileNames[i]+"Predictions"+j+".csv");
                of2.writeLine(inF.readLine());
                inF.readLine();
                for(int k=0;k<lines;k++){
                    int[] d=new int[2];
                    d[0]=inF.readInt();
                    d[1]=inF.readInt();
                    preds.add(d);
                }
            }
//Load ordering
            int[] orders=new int[preds.size()];
            inF=new InFile(path+DataSets.fileNames[i]+"/ShapeletCV/InstancePositions.csv"); 
            lines=inF.countLines();
            if(lines!=preds.size()){ //ERROR
                System.err.println(" BIG ERROR: reording number does not equal the number of cases in the file!!! Problem ="+DataSets.fileNames[i]);
                System.err.println(" \t\t in recorded positions there are" +lines+" in the combo results there are "+preds.size());
                continue fileLoop;  
            }
            inF=new InFile(path+DataSets.fileNames[i]+"/ShapeletCV/InstancePositions.csv"); 
            for(int k=0;k<lines;k++)
                orders[k]=inF.readInt();
//Reorder into original //Work out Cv Train Accuracy        
            int[][] results=new int[lines][];
            int correct=0;
            for(int k=0;k<lines;k++){
                results[orders[k]]=preds.get(k);
                if(results[orders[k]][0]==results[orders[k]][1])
                    correct++;
            }
//Print to file
            for(int k=0;k<lines;k++)
                of.writeLine(results[k][0]+","+results[k][0]);
            all.writeString(","+((double)correct)/lines);
        }
    }
    
    public static ArrayList<Classifier> setSingleClassifiers(ArrayList<String> names){
            ArrayList<Classifier> sc=new ArrayList<>();
            kNN n= new kNN(50);
            n.setCrossValidate(true);
            sc.add(n);
            names.add("kNN");
            sc.add(new J48());
            names.add("C45");
            sc.add(new NaiveBayes());
            names.add("NB");
            BayesNet bn = new BayesNet();
            sc.add(bn);
            names.add("BayesNet");
            RandomForest rf = new RandomForest();
            rf.setNumTrees(200);
            sc.add(rf);
            names.add("RandForest");
            RotationForest rot = new RotationForest();
            rot.setNumIterations(30);
            sc.add(rf);
            names.add("RotForest");
            SMO svmL = new SMO();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            svmL.setKernel(kernel);
            sc.add(svmL);
            names.add("SVML");
            kernel = new PolyKernel();
            kernel.setExponent(2);
            SMO svmQ = new SMO();
            svmQ.setKernel(kernel);   
            sc.add(svmQ);
            names.add("SVMQ");
            return sc;
	}    

    public static void doTransform(String[] args){
//        checkTrainsforms();
//        System.exit(0);
        
        if(args.length==0){
            useCluster=false;
            System.out.println(" ON DESKTOP");
            int pos=1;
                System.out.println(" Transforming :"+DataSets.fileNames[34]);
                singleRunThreaded("ItalyPowerDemand");
            
        }
        
        else{
            useCluster=true;
            int num=Integer.parseInt(args[0]);
            int problemNum=num-1;
            System.out.println(" Transforming ="+DataSets.fileNames[problemNum]);
            singleRunThreaded(DataSets.fileNames[problemNum]);
                
        }
         
    }
    
    
    
    public static void classifyProblem(String[] args){
        if(args.length==0){
            useCluster=false;
            System.out.println(" ON DESKTOP");
            int pos=1;
 //               System.out.println(" Classifying :"+DataSets.fileNames[pos]);
                for(int i=1;i<=10;i++){
                    int[][] res=classifyFold("ItalyPowerDemand",i);
                }
        }
        else{
            useCluster=true;
            int n=Integer.parseInt(args[0])-1;
            int problemNum=n/10;
            int foldNum=n%10;
//Results saved to individual files 
            int[][] res=classifyFold(DataSets.fileNames[problemNum],foldNum+1);
           
               
        }       
    }
    
    public static void purge(){ //Delete all CV files from the cluster
        useCluster=true;
        for(int i=0;i<DataSets.fileNames.length;i++)
        {
            String clusterPath = "/gpfs/sys/ajb/TSC Problems/"+DataSets.fileNames[i]+"/";
            String desktopPath="C:/Users/ajb/Dropbox/TSC Problems/"+DataSets.fileNames[i]+"/";
            String path=desktopPath;
            if(useCluster)
                path=clusterPath;
            File f = new File(path+"ShapeletCV/");
            if(f.exists()){
                try{
                    delete(f);
                }catch(IOException e){
                    System.err.println(" Unable to delete directory "+path+"ShapeletCV/  Continuing  ");
                }
            }
        }
    }
    public static void delete(File file) throws IOException{ 
    	if(file.isDirectory()){
    		//directory is empty, then delete it
    		if(file.list().length==0){
    		   file.delete();
    		   System.out.println("Directory is deleted : " 
                                                 + file.getAbsolutePath());
    		}else{
    		   //list all the directory contents
        	   String files[] = file.list();
        	   for (String temp : files) {
        	      //construct the file structure
        	      File fileDelete = new File(file, temp);
        	      //recursive delete
        	     delete(fileDelete);
        	   }
 
        	   //check the directory again, if empty then delete it
        	   if(file.list().length==0){
           	     file.delete();
        	     System.out.println("Directory is deleted : " 
                                                  + file.getAbsolutePath());
        	   }
    		}
 
    	}else{  //Base case
    		//if file, then delete it
    		file.delete();
    		System.out.println("File is deleted : " + file.getAbsolutePath());
    	}
    }    
    public static void transformIncomplete(String[] args){
        int length=incomplete.length;    //12 of these
        int n=Integer.parseInt(args[0])-1;
        int problemNum=n/10;
        int foldNum=n%10;
        if(problemNum>=length)  //Error
            return;
        problemNum=incomplete[problemNum];
        doSingleTransform(problemNum,foldNum);
    } 
    public static void doSingleTransform(int problemNum,int foldNum){
        String fileName=DataSets.fileNames[problemNum];
        String clusterPath = "/gpfs/sys/ajb/TSC Problems/"+fileName+"/";
        String path=clusterPath;
        String shapeletPath=path+"ShapeletCV/";
        File f1=  new File(shapeletPath+fileName+"_TRAIN"+(foldNum+1)+".arff");
        File f2 =  new File(shapeletPath+fileName+"_TEST"+(foldNum+1)+".arff");
        if(f1.exists() && f2.exists()){
            System.out.println(" Transform "+foldNum+" problem "+fileName+" already exists");
            return;
        }
        
        
        Instances train = ClassifierTools.loadData(clusterPath+fileName+"_TRAIN"+(foldNum+1));
        Instances test = ClassifierTools.loadData(clusterPath+fileName+"_TEST"+(foldNum+1));
        FullShapeletTransform st=new ShapeletTransformDistCaching();
//        if(train.numInstances()>=500 || train.numAttributes()>500)
//            st = new ShapeletTransform();
        st.supressOutput();
        st.setNumberOfShapelets(Math.max(train.numAttributes(), train.numInstances()));
         try {
            Instances sTrain=st.process(train);
            Instances sTest=st.process(test);
            OutFile of1 =  new OutFile(shapeletPath+fileName+"_TRAIN"+(foldNum+1)+".arff");
            OutFile of2 =  new OutFile(shapeletPath+fileName+"_TEST"+(foldNum+1)+".arff");
            of1.writeLine(sTrain.toString());
            of2.writeLine(sTest.toString());
            
        } catch (Exception ex) {
            Logger.getLogger(CrossValidateShapelets.class.getName()).log(Level.SEVERE, null, ex);
        }
       
    } 
    public static void shapeletTrainSingle(String file){
        String clusterPath = "/gpfs/sys/ajb/ShapeletTransformed/";
        String desktopPath="C:/Users/ajb/Dropbox/TSC Problems/ShapeletTransformed/";
        String path=desktopPath;
        if(useCluster)
            path=clusterPath;
 //Load 
        OutFile of=new OutFile(path+"TrainCV/"+file+"_trainCVacc.csv");
        File f= new File(path+file+"Transformed_TRAIN");
        if(!f.exists()){
            of.writeLine(file+","+"-1");
        }
        Instances train = ClassifierTools.loadData(path+file+"Transformed_TRAIN");
//Get classifiers
             ArrayList<String> names= new ArrayList<>();
        ArrayList<Classifier> c=setSingleClassifiers(names);
        HeterogeneousEnsemble hc = new HeterogeneousEnsemble(c);
        hc.useCVWeighting(true);
//Find Accuracy        
        double acc=ClassifierTools.stratifiedCrossValidation(train, hc,10, 1);
//Get individual stats
//Write to file
        of.writeLine(file+","+acc);
        
        
    }
    public static void shapeletTrainCV(String[] args){
        
        if(args.length==0){
            useCluster=false;
            System.out.println(" ON DESKTOP");
            int pos=1;
                System.out.println(" Transforming :"+DataSets.fileNames[34]);
                shapeletTrainSingle("ItalyPowerDemand");
            
        }
        
        else{
            useCluster=true;
            int num=Integer.parseInt(args[0]);
            int problemNum=num-1;
            System.out.println(" Transforming ="+DataSets.fileNames[problemNum]);
            shapeletTrainSingle(DataSets.fileNames[problemNum]);
                
        }
         
    }
    
    
    
    public static void combineShapeletTrain(){
        String path="C:/Users/ajb/Dropbox/Results/ShapeletDomain/TrainCV/";
        OutFile combo=new OutFile(path+"allResults.csv");
        for(String s:DataSets.fileNames){
            File f= new File(path+s+"_trainCVacc.csv");
            if(!f.exists()){
                combo.writeLine(s+",");
                System.out.println(path+s+"_trainCVacc.csv"+" DOES NOT EXIST");
            }
            else{
                InFile inf = new InFile(path+s+"_trainCVacc.csv");
                String str=inf.readLine();
                combo.writeLine(str+","+inf.readLine());
            }
        }
    }
    public static void main(String[] args){
//        formCV();
//        transformIncomplete(args);
//        doTransform(args);
//           classifyProblem(args);
//                combineandInvertFolds(); 
        
//        shapeletTrainCV(args);
        combineShapeletTrain();
   }    
    
}
