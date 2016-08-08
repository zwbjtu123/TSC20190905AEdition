/*
Combines various files into standard results format.

Levels of collation at problem, classifier and split

1. single file by problem, classifier and split to single file of problem and classifier 
NaiveBayes/Adiac1.csv ....NaiveBayes/Adiac1.csv

to 
NaiveBayes/Adiac.csv
Adiac,0.6,0.53,0.55...

2. single file of problem and classifier  to single file of classifier
NaiveBayes/Adiac.csv
NaiveBayes/ArrowHead.csv

to 
NaiveBayes/NaiveBayes.csv
Where collated data is in format
problem, mean, std dev, sample size


3. single file for each classifier into single file
Results/NaiveBayes/NaiveBayes.csv
Results/NaiveBayes/C45.csv
to
Results/Accuracy.csv
Results/StDev.csv
Results/SampleSize.csv



*/
package new_COTE_experiments;

import UCI_classification.UCIClassification;
import bakeOffExperiments.*;
import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class FormatHiveCOTE {
   static int[] testSizes={391,175,30,20,20,60,900,3840,1380,28,250,390,390,390,306,276,139,139,139,100,4500,861,7711,1690,88,2050,455,175,1320,810,150,105,370,308,64,550,1980,1029,375,61,73,2345,60,760,291,154,154,1252,1965,1965,30,242,858,1896,105,291,205,205,375,375,180,600,375,601,953,8236,370,625,995,300,228,130,100,1139,4000,3582,3582,3582,3582,6164,54,638,77,77,3000};
    static String[] classifiers={"PS_ACF","RIF_PS","RIF_ACF","RIF_PS_ACF"};//
//    static String[] classifiers={"WE","BOSS","TSF","RIF_PS","RIF_ACF","RIF_PS_ACF"};
    HashSet<String> finished=new HashSet<>();

/** 
Generate all the scripts for a single algorithm 

* */
    public static void generateAllScripts(String path, String classifier, boolean oldCls, int mem){
      
        int maxMem=mem+1000;
        String queue,java; 
        if(oldCls){
            queue="long";
            java= "java/jdk/1.8.0_31";
        }
        else{
            queue="long-eth";
            java="java/jdk1.8.0_51";
        }
        File f=new File(path+"/"+classifier);
        deleteDirectory(f);
        if(!f.isDirectory())
            f.mkdir();
        for(int i=0;i<DataSets.fiveSplits.length;i++){
            OutFile of2;
            if(oldCls)
                of2=new OutFile(path+"/"+classifier+(i+1)+"OldCls.txt");
            else
                of2=new OutFile(path+"/"+classifier+(i+1)+".txt");
            for(int j=0;j<DataSets.fiveSplits[i].length;j++){
                String prob=DataSets.fiveSplits[i][j];
                OutFile of;
                if(oldCls)
                    of=new OutFile(path+"/"+classifier+"/"+prob+"OldCls.bsub");
                else
                    of=new OutFile(path+"/"+classifier+"/"+prob+".bsub");
                of.writeString("#!/bin/csh\n" +
                "#BSUB -q ");
                of.writeString(queue+"\n#BSUB -J ");
                of.writeLine(classifier+prob+"[1-100]");
                of.writeString("#BSUB -oo output/"+classifier+prob+"%I.out\n" +
                    "#BSUB -eo error/"+classifier+prob+"%I.err\n" +
                    "#BSUB -R \"rusage[mem="+mem+"]\"\n" +
                    "#BSUB -M "+maxMem);
                of.writeLine("\n\n module add "+java);

                of.writeLine("java -jar HiveCOTE.jar "+classifier+" "+prob+" $LSB_JOBINDEX ");
                if(oldCls)
                    of2.writeLine("bsub < "+"Scripts/"+classifier+"/"+prob+"OldCls.bsub");                
                else
                    of2.writeLine("bsub < "+"Scripts/"+classifier+"/"+prob+".bsub");
                of.closeFile();
            }
        }
    }
    
/* Step 2: merge single problem files for classifier into one file
    */    
    public static void combineSingleProblemFilesIntoOneFile(String path, String classifier){
        File dir= new File(path+"\\"+classifier);
        if(dir.isDirectory()){    //Proceed if there is a directory of results
            OutFile results = new OutFile(path+"\\"+classifier+".csv");        
            for(String s:DataSets.fileNames){
                 File f= new File(path+"\\"+classifier+"\\"+s+".csv");
                 if(f.exists()){
                     InFile f2=new InFile(path+"\\"+classifier+"\\"+s+".csv");
                     String str=f2.readLine();
                     results.writeLine(str);
                 }
             }
            results.closeFile();
        }
    }
 
/* Takes a possibly partial list of results and format into the outfile */
    public static class Results{
        String name;
        double mean;
        double median;
        double stdDev;
        double[] accs;
        public boolean equals(Object o){
            if( ((Results)o).name.equals(this.name))
                return true;
            return false;
        }
    }
 
    public static void collateDifferenceBetweenTrainTest(String results, String classifier, String[] fileNames){
        OutFile out=new OutFile(results+classifier+"/TrainTestCombined.csv");
//Final ones
        for(String s: fileNames){
            double mean=0;
            double count=0;
            double sumSq=0;
            for(int i=0;i<100;i++){
                if(new File(results+classifier+"/Predictions/"+s+"/testFold"+i+".csv").exists() &&
                    new File(results+classifier+"/Predictions/"+s+"/trainFold"+i+".csv").exists()   ){
                    try{
                        InFile test=new InFile(results+classifier+"/Predictions/"+s+"/testFold"+i+".csv");
                        InFile train=new InFile(results+classifier+"/Predictions/"+s+"/trainFold"+i+".csv");
                        test.readLine();
                        train.readLine();
                        test.readLine();
                        train.readLine();
                        double a =train.readDouble();
                        double b=test.readDouble();
                        mean+=(a-b);
                        sumSq+=(a-b)*(a-b);
                        count++;
                    }catch(Exception e){
                    System.out.println("Unable to read "+results+classifier+"/Predictions/"+s+"/testFold"+i+".csv");
                        
                    }
                }
                else{
                    System.out.println("DOESBOT exist "+results+classifier+"/Predictions/"+s+"/testFold"+i+".csv");
                }
            }
            mean/=count;
            out.writeLine(s+","+count+","+mean+","+sumSq);
        }
        
    }
    public static void incompleteResultsParser(InFile f, OutFile of) throws Exception{
        int lines= f.countLines();
        f.reopen();
        ArrayList<Results> res=new ArrayList<>();
  //      System.out.println("Lines = "+lines);
        
        for(int i=0;i<lines;i++){
            String line=f.readLine();
            String[] split=line.split(",");
            Results r=new Results();
            r.mean=0;
            r.name=split[0];
            ArrayList<Double> accs=new ArrayList<>();
            int count=0;
            if(split.length>1){
    //            System.out.println("length="+r.accs.length+"::::"+line);
                //May have internal ones missing
                for(int j=0;j<split.length-1;j++){
                    try{
                        if(!split[j+1].equals(""))
                            accs.add(Double.parseDouble(split[j+1]));
                    }catch(Exception e){
                        System.out.println("ERROR: "+split[j]+" giving error "+e+" in file "+f.getName()+" on line "+i+" name ="+r.name);
                        System.exit(0);
                    }
                }
                r.accs=new double[accs.size()];
                for(int j=0;j<accs.size();j++){
                    r.accs[j]=accs.get(j);
                    r.mean+=r.accs[j];
                }
                r.mean/=r.accs.length;
                r.stdDev=0;
                for(int j=0;j<r.accs.length;j++){
                    r.stdDev+=(r.accs[j]-r.mean)*(r.accs[j]-r.mean);
                }
                r.stdDev/=(r.accs.length-1);
                r.stdDev=Math.sqrt(r.stdDev);
                Arrays.sort(r.accs);
                if(r.accs.length%2==0)
                    r.median=(r.accs[r.accs.length/2]+r.accs[r.accs.length/2-1])/2;
                else
                    r.median=r.accs[r.accs.length/2];
                    
            }
            res.add(r);
        }
        for(int i=0;i<DataSets.fileNames.length;i++){
            of.writeString(DataSets.fileNames[i]+",");
            int j=0; //Wasteful linear scan
            while(j<res.size() && !DataSets.fileNames[i].equals(res.get(j).name))
                j++;
//            System.out.println("J =: "+j+" "+res.size());
            if(j<res.size()){
                Results r=res.get(j);
                if(r.mean>0)
                    of.writeLine(r.mean+","+r.stdDev+","+r.accs.length+","+r.median);
                else
                    of.writeLine("");
            }
            else
                of.writeLine("");
        }
    }    
    

    public static void findStatsAndCombineClassifiers(String inPath,String outPath) throws Exception{
        OutFile[] of={new OutFile(outPath+"\\Means.csv"),new OutFile(outPath+"\\StDevs.csv"),new OutFile(outPath+"\\SampleSizes.csv"),new OutFile(outPath+"\\Medians.csv")};
        
        for(OutFile o:of){
            o.writeString(",");
            for(String st:classifiers)
                o.writeString(st+",");
        }
        for(OutFile o:of)
            o.writeString("\n");
        
//Try open all
        InFile[] inFiles=new InFile[classifiers.length];
        for(int i=0;i<inFiles.length;i++){
//Check existence
            File f=new File(inPath+classifiers[i]+".csv");
//If exists, open                
            if(f.exists())
                inFiles[i]=new InFile(inPath+classifiers[i]+".csv");
            else{
                inFiles[i]=null;
//                    System.out.println(" File "+classifiers[i][j]+" does not exist");
            }
        }
        for(String s:DataSets.fileNames){
            for(OutFile o:of)
                o.writeString(s+",");
            for(int i=0;i<inFiles.length;i++){
                if(inFiles[i]==null){
                        of[0].writeString(",");
                        of[1].writeString(",");
                        of[2].writeString("0,");
                        of[3].writeString(",");
                }
                else{
                    String[] name=inFiles[i].readLine().split(",");
                    if(name.length<2){
                        of[0].writeString(",");
                        of[1].writeString(",");
                        of[2].writeString("0,");
                        of[3].writeString(",");
                    }
                    else{
                        for(int k=0;k<of.length;k++)
                            of[k].writeString(name[k+1]+",");
                    }
                }
          }    
            for(OutFile o:of)
               o.writeString("\n");
        }
    }

    
    
    
    /**
     * Go from individual folds files in this format: single classifier and problem
ProblemFile,ClassifierName, test
ParameterInfo
TestAccuracy
Predictions with probabilities (which can be ignored here)
     * @param source
     * @param dest
     * @param problemID
     * @throws java.lang.Exception
    **/
    public static void combineFolds(String source, OutFile dest, int problemID) throws Exception{
        for(int i=0;i<100;i++){
            File inf=new File(source+"testFold"+i+".csv");
//            System.out.println(" Reading "+inf.getPath());
            if(inf.exists() && inf.length()>0){
                InFile f=new InFile(source+"testFold"+i+".csv");
                int lines=f.countLines();
                int testCount=testSizes[problemID]-3;
                if(lines>testCount){//Ignore incomplete
                    f=new InFile(source+"testFold"+i+".csv");
                    f.readLine();
                    f.readLine();
                    double acc=f.readDouble();
                    if(i==0)
                        System.out.println(source+" Test Acc = "+acc);
                    if(i<99)
                        dest.writeString(acc+",");
                    else
                        dest.writeLine(acc+"");
                }  
                else{
                    System.out.println(inf.getPath()+" not complete only "+lines+" cases instead of "+testSizes[problemID]);
                }
            }
        }
    }
    
    public static void checkTrainTestData(String root) throws Exception{
        
        for(int i=0;i<classifiers.length;i++)
        {
            OutFile of=new OutFile(root+"\\"+classifiers[i]+"missingTrainTestFolds.csv"); 
//Check for directory of
            File dir= new File(root+"\\"+classifiers[i]+"\\");
            if(dir.isDirectory()){    //Proceed if there is a directory of results
                for(int k=0;k<DataSets.fileNames.length;k++){
                    String s= DataSets.fileNames[k];
                    of.writeString(s+",");
                    dir= new File(root+"\\"+classifiers[i]+"\\Predictions"+"\\"+s);
//Check if there is a directory of predictions.
                    if(dir.isDirectory()){
//                            int temp=checkPredictionLength(k,dir.getPath(),out);
//                            numFolds+=temp;
//                           if(temp==100)
//                               completeCount++;
                //The files dir+"\\"+s+".csv" contain the average accuracy per fold. 
                        String p=root+"\\"+classifiers[i];
                        int testCount=0;
                        int trainCount=0;
                        for(int j=0;j<100;j++){
                            File y=new File(p+"\\Predictions\\"+s+"\\"+"testFold"+j+".csv"); 
                            if(y.exists() && y.length()>0){
//Count the lines
                                    testCount++;
/*                                InFile inf=new InFile(p+"\\Predictions\\"+s+"\\"+"testFold"+j+".csv");
                                int lines=inf.countLines();
                                if(lines-3==testSizes[i])
                                    testCount++;
                                else
                                    System.out.println(classifiers[i]+" "+s+" incomplete test fold "+j);
  */                          }
                            else
                                System.out.println(classifiers[i]+" "+s+" missing test fold "+j);
                        }
                        for(int j=0;j<100;j++){
                            File x=new File(p+"\\Predictions\\"+s+"\\"+"trainFold"+j+".csv");
                            if(x.exists() && x.length()>0){
//Count the lines
//                                InFile inf=new InFile(p+"\\Predictions\\"+s+"\\"+"trainFold"+j+".csv");
//                                int lines=inf.countLines();
                                trainCount++; //Should check the sizes
                            }
                            else
                                System.out.println(classifiers[i]+" "+s+" missing train fold "+j);
                        }
                        of.writeString(trainCount+","+testCount);
                    }
                    of.writeLine("");
                }
            }
            else{
                System.out.println(root+"\\"+classifiers[i]+"\\ does not exist");
            }
        }
    }
    
    
   
    
/**
 * Step 1: go from predictions files into problem files using combineFolds
 * @param root
 * @throws Exception 
 */    
    public static void combineIndividualFoldsIntoProblemFiles(String root) throws Exception{
        for(int i=0;i<classifiers.length;i++)
        {
//Check for directory of
            File dir= new File(root+"\\"+classifiers[i]+"\\");
            if(dir.isDirectory()){    //Proceed if there is a directory of results
                for(int k=0;k<DataSets.fileNames.length;k++){
                    String s= DataSets.fileNames[k];
                    dir= new File(root+"\\"+classifiers[i]+"\\Predictions"+"\\"+s);
//Check if there is a directory of predictions.
                  if(dir.isDirectory()){
//                            int temp=checkPredictionLength(k,dir.getPath(),out);
//                            numFolds+=temp;
//                           if(temp==100)
//                               completeCount++;
                //The files dir+"\\"+s+".csv" contain the average accuracy per fold. 
                        String p=root+"\\"+classifiers[i];
                        File f=new File(p+"\\"+s+".csv");
                            //Check if there are any predictions
                        if(checkPredictions(p+"\\Predictions\\"+s+"\\")){
                            OutFile of=new OutFile(p+"\\"+s+".csv"); 
                            of.writeString(s+",");
                            combineFolds(p+"\\Predictions\\"+s+"\\",of,k);
                            of.closeFile();
                        }
                    }
                }
            }
            else{
                System.out.println(root+"\\"+classifiers[i]+"\\ does not exist");
            }
        }
    }
    
    
    
    
    
    
    
    public static boolean checkPredictions(String path){
        for(int i=0;i<100;i++){
            if(new File(path+"testFold"+i+".csv").exists())
                return true;
        }
        return false;
    }


    public static int checkPredictionLength(int problem, String path, OutFile of){
        System.out.println("Checking "+path);
        int count=0;
        boolean complete=true;
        for(int i=0;i<100;i++){
            File f2=new File(path+"\\fold"+i+".csv");
            if(f2.exists()){
                InFile f=new InFile(path+"\\fold"+i+".csv");
                int lines=f.countLines();
                if(lines!=testSizes[problem]){
                    of.writeLine(path+","+i+","+lines+","+testSizes[problem]);
//                    System.out.println("INCOMPLETE FOLD :"+path+","+i+","+lines+","+testSizes[problem]);
                    if(lines==0){
                        f.closeFile();
                        f2.delete();
                    }
                }
                else
                    count++;
            }
        }
        return count;
    }
    public static void generateParameterSplitScripts(String root, String dest,String classifier,String problem,int paras,int folds){
        InFile inf=new InFile(root+"\\SampleSizes.csv");
        File f=new File(dest+"\\Scripts\\"+problem);
        deleteDirectory(f);
        if(!f.isDirectory())
            f.mkdir();
        OutFile of2=new OutFile(dest+"\\Scripts\\"+problem+"\\"+problem+"paras.txt");
        for(int j=1;j<=paras;j++){
            OutFile of=new OutFile(dest+"\\Scripts\\"+problem+"\\paraFold"+"_"+j+".bsub");
            of.writeString("#!/bin/csh\n" +
                "#BSUB -q ");
            of.writeString("long\n#BSUB -J ");
            of.writeLine(classifier+problem+"[1-10]");
            of.writeString("#BSUB -oo output/"+classifier+"%I.out\n" +
              "#BSUB -eo error/"+classifier+"%I.err\n" +
              "#BSUB -R \"rusage[mem=2000]\"\n" +
              "#BSUB -M 3000");
            of.writeLine("\n\n module add java/jdk/1.8.0_31");
            of.writeLine("java -jar HiveCOTE.jar "+classifier+" " +problem+" $LSB_JOBINDEX"+j);
            of2.writeLine("bsub < "+"Scripts/"+classifier+"/Unstarted/"+problem+".bsub");
            of.closeFile();
        }
        
    }
    public static void fileStandardiseForProblems(String path) throws Exception{
       for(String s:classifiers){
            File f= new File(path+"\\"+s+".csv");
            if(f.exists()){
                InFile inf=new InFile(path+"\\"+s+".csv");
                OutFile outf=new OutFile(path+"\\Collated\\"+s+".csv");
                incompleteResultsParser(inf,outf);
                inf.closeFile();
                outf.closeFile();
            }
            else
                System.out.println(" File "+path+"\\"+classifiers+"\\"+s+" does not exist");
        }
    }    
    
    public static boolean deleteDirectory(File directory) {
        if(directory.exists()){
            File[] files = directory.listFiles();
            if(null!=files){
                for(int i=0; i<files.length; i++) {
                    if(files[i].isDirectory()) {
                        deleteDirectory(files[i]);
                    }
                    else {
                        files[i].delete();
                    }
                }
            }
        }
        return(directory.delete());
    }    
    public static void main(String[] args) throws Exception{
        String root="C:\\Users\\ajb\\Dropbox\\NewCOTEResults\\Spectral\\";
  //      generateAllScripts(root+"Scripts","PS_ACF",false,18000);
 //   System.exit(0);
 //       collateDifferenceBetweenTrainTest("C:\\Users\\ajb\\Dropbox\\NewCOTEResults\\", "RIF_PS_ACF", DataSets.fileNames);

//    
//        checkTrainTestData(root);
//        generateAllScripts(root+"Scripts","RIF_PS_ACF",false,8000);
//    System.exit(0);
//    generateAllScripts("C:\\Users\\ajb\\Dropbox\\NewCOTEResults\\Scripts\\","TSF",false,6000);
      DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\NewCOTEResults\\Spectral\\";
        System.out.println("Step 1: combine prediction files into single file for each classifier ....");
        combineIndividualFoldsIntoProblemFiles(root);
        System.out.println(" Step 2: merge single problem files for classifier into one file...");
        for(String cls:classifiers)
            combineSingleProblemFilesIntoOneFile(root,cls);
          System.out.println("Step 3: standardise file formats to allow for missing problems ....");
           fileStandardiseForProblems(root);
        System.out.println("Step 4: Find stats and write to a single file ....");
           findStatsAndCombineClassifiers(root+"\\Collated\\",root);
        
//        generateScripts(root,root);

    }
    public static void findNumberPerSplit(){
        String path="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\ensemble\\PS\\Predictions\\";
        int min;
            System.out.print("{");
        for(int i=0;i<DataSets.fileNames.length;i++){
            min=Integer.MAX_VALUE;
            int max=0;
            for(int j=0;j<100;j++){
                InFile f= new InFile(path+DataSets.fileNames[i]+"\\fold"+j+".csv");
                int t=f.countLines();
                if(t<min)
                    min=t;
                if(t>max)
                    max=t;
            }
            System.out.print(max+",");
        }
        System.out.print("};");
    }

    
    public static void formatHeartbeatBIDMC(){
        String path="C:\\Users\\ajb\\Dropbox\\TSC Problems\\HeartbeatBIDMC";
        InFile f=new InFile(path+"\\HeartbeatBIDMC_TEST.txt");
        int numAtts=3750;
        int numInst=600;
        OutFile of=new OutFile(path+"\\HeartbeatBIDMC_TEST.arff");
        of.writeLine("@relation HeartbeatBIDMC");
        of.writeLine("");
        for(int i=0;i<numAtts;i++){
            of.writeLine("@attribute att"+i+" numeric");
        }
        of.writeString("@attribute personID {");
        for(int i=1;i<=14;i++)
            of.writeString(i+",");
        of.writeLine("15}");
        of.writeLine("\n @data");
        for(int i=0;i<numInst;i++){
            int classV=f.readInt();
            for(int k=0;k<numAtts;k++){
                double x=f.readDouble();
                of.writeString(x+",");
            }
            of.writeLine(classV+"");
            f.readLine();
        }
    }
}
