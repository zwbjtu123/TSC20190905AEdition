/*
class to collate standard results files over multiple classifiers and problems

Usage: 



 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import utilities.ClassifierResults;

/**
 *
 * @author ajb
 */
public class CollateResults {
    public static File[] dir;
    static String basePath;
    static String[] classifiers;
    static ArrayList<String> problems;
    static String problemPath;
    static int folds;
    static int numClassifiers;
    static int[] numParas;
    static DecimalFormat df=new DecimalFormat("##.######");
    
  
    public static void readData(String[] args){
        basePath=args[0];
        System.out.println("Base path = "+basePath);
        problemPath=args[1];
        System.out.println("Problem path = "+problemPath);
         folds =Integer.parseInt(args[2]);
        System.out.println("Folds = "+folds);
        numClassifiers=(args.length-3)/2;
        classifiers=new String[numClassifiers];
        for(int i=0;i<classifiers.length;i++)
            classifiers[i]=args[i+3];
        numParas=new int[classifiers.length];
        for(int i=0;i<classifiers.length;i++)
            numParas[i]=Integer.parseInt(args[i+3+classifiers.length]);
//Get problem files
        File f=new File(problemPath);
        if(!f.isDirectory()){
            System.out.println("Error in problem path ="+problemPath);
        }
            
        dir=f.listFiles();
        problems=new ArrayList<>();
        for(File p:dir){
            if(p.isDirectory()){
                problems.add(p.getName());
            }
        }
        Collections.sort(problems);
        
    }    
    
    public static void collateFolds(){

        for(int i=0;i<classifiers.length;i++){
            String cls=classifiers[i];
            System.out.println("Processing classifier ="+cls);
            File f=new File(basePath+cls);
            if(f.isDirectory()){ //Check classifier directory exists. 
//Write collated results for this classifier to a single file                
                OutFile clsResults=new OutFile(basePath+cls+"/"+cls+"TestAcc.csv");
                OutFile f1Results=new OutFile(basePath+cls+"/"+cls+"TestF1.csv");
                OutFile BAccResults=new OutFile(basePath+cls+"/"+cls+"TestBAcc.csv");
                System.out.println(basePath+cls+"/"+cls+"TestAcc.csv");
                OutFile trainResults=new OutFile(basePath+cls+"/"+cls+"TrainCVAcc.csv");
                OutFile[] paraFiles=new OutFile[numParas[i]];
                for(int j=0;j<paraFiles.length;j++)
                    paraFiles[j]=new OutFile(basePath+cls+"/"+cls+"Parameter"+(j+1)+".csv");
                OutFile timings=new OutFile(basePath+cls+"/"+cls+"Timings.csv");
                OutFile allAccSearchValues=new OutFile(basePath+cls+"/"+cls+"AllTuningAccuracies.csv");
                OutFile missing=null;
                OutFile counts=new OutFile(basePath+cls+"/"+cls+"Counts.csv");;
                
                int missingCount=0;
                for(String name:problems){            
                    clsResults.writeString(name+",");
                    trainResults.writeString(name+",");
                    f1Results.writeString(name);
                    BAccResults.writeString(name);
                    for(OutFile out:paraFiles)
                        out.writeString(name+",");
                    String path=basePath+cls+"//Predictions//"+name;
                    if(missing!=null && missingCount>0)
                        missing.writeString("\n");
                    missingCount=0;
                    for(int j=0;j<folds;j++){
    //Check fold exists
                        f=new File(path+"//testFold"+j+".csv");

                        if(f.exists() && f.length()>0){//This could fail if file only has partial probabilities on the line
    //Read in test accuracy and store                    
    //Check fold exists
    //Read in test accuracy and store
                            InFile inf=null;
                            String[] trainRes=null;
                            try{
                                inf=new InFile(path+"//testFold"+j+".csv");
                                inf.readLine();
                                trainRes=inf.readLine().split(",");//Stores train CV and parameter info
                                clsResults.writeString(inf.readDouble()+",");
                                if(trainRes.length>1){//There IS parameter info
                                    //First is train time build
                                    timings.writeString(Double.parseDouble(trainRes[1])+",");
                                    //second is the trainCV testAcc
                                    if(trainRes.length>3){
                                        trainResults.writeString(Double.parseDouble(trainRes[3])+",");
                                        //Then variable list of numParas
                                        int pos=5;
                                        for(int k=0;k<numParas[i];k++){
                                            if(trainRes.length>pos){
                                                paraFiles[k].writeString(Double.parseDouble(trainRes[pos])+",");
                                                pos+=2;    
                                            }
                                            else
                                                paraFiles[k].writeString(",");
                                        }
    //                                    write the rest to the para search file
                                        while(pos<trainRes.length)
                                            allAccSearchValues.writeString(trainRes[pos++]+",");    
                                    allAccSearchValues.writeString("\n");

                                    }
                                }
                                else{
                                    trainResults.writeString(",");
                                    for(int k=0;k<numParas[i];k++)
                                        paraFiles[k].writeString(",");
                                }
//Read in the rest into a ClassifierResults object
                                inf.closeFile();
                                ClassifierResults res=new ClassifierResults();
                                res.loadFromFile(path+"//testFold"+j+".csv");
                                res.findAllStats();
                                f1Results.writeString(","+res.f1);
                                BAccResults.writeString(","+res.balancedAcc);
                                
                            }catch(Exception e){
                                System.out.println(" Error "+e+" in "+path);
                                if(trainRes!=null){
                                    System.out.println(" second line read has "+trainRes.length+" entries :");
                                    for(String str:trainRes)
                                        System.out.println(str);
                                }
                                System.exit(1);
                            }finally{
                                if(inf!=null)
                                    inf.closeFile();

                            }
                        }
                        else{
                            if(missing==null)
                                missing=new OutFile(basePath+cls+"//"+cls+"MissingFolds.csv");
                            if(missingCount==0)
                                missing.writeString(name);
                            missingCount++;
                           missing.writeString(","+j);
                        }
                    }
                    counts.writeLine(name+","+(folds-missingCount));
                    clsResults.writeString("\n");
                    trainResults.writeString("\n");
                    for(int k=0;k<paraFiles.length;k++)
                        paraFiles[k].writeString("\n");
                }
                clsResults.closeFile();
                trainResults.closeFile();
                    for(int k=0;k<paraFiles.length;k++)
                        paraFiles[k].closeFile();
            }
            else
                System.out.println("Classifier "+cls+" has no results directory: "+basePath+cls);
        }
        
    }

    public static void averageOverFolds(){
//3. Merge classifier files into files with
//Counts for all
// mean test and trainCV accuracies        
//std dev test and trainCV accuracies        
//mean difference in train-test accuracies        
//std dev difference in train-test accuracies
//mean F1 statistic
//Mean balanced error        
        
        String name=classifiers[0];
        for(int i=1;i<classifiers.length;i++)
            name+=classifiers[i];
        OutFile testAcc=new OutFile(basePath+"TestAcc"+name+".csv");
        OutFile testStdDev=new OutFile(basePath+"TestStdDev"+name+".csv");
        OutFile trainCVAcc=new OutFile(basePath+"TrainCVAcc"+name+".csv");
        OutFile trainCVStdDev=new OutFile(basePath+"TrainCVStdDev"+name+".csv");
        OutFile diffAcc=new OutFile(basePath+"DiffAcc"+name+".csv");
        OutFile diffStdDev=new OutFile(basePath+"DiffStdDev"+name+".csv");
        OutFile count=new OutFile(basePath+"Count"+name+".csv");
        for(int i=0;i<classifiers.length;i++){
            String cls=classifiers[i];
            testAcc.writeString(","+cls);
            testStdDev.writeString(","+cls);
            trainCVAcc.writeString(","+cls);
            trainCVStdDev.writeString(","+cls);
            diffAcc.writeString(","+cls);
            diffStdDev.writeString(","+cls);
            count.writeString(","+cls);
        }
        testAcc.writeString("\n");
        testStdDev.writeString("\n");
        trainCVAcc.writeString("\n");
        trainCVStdDev.writeString("\n");
        diffAcc.writeString("\n");
        diffStdDev.writeString("\n");
        count.writeString("\n");
        InFile[] allTest=new InFile[classifiers.length];
        InFile[] allTrainCV=new InFile[classifiers.length];
        for(int i=0;i<allTest.length;i++){
            String p=basePath+classifiers[i]+"/"+classifiers[i]+"TestAcc.csv";
            if(new File(p).exists())
                allTest[i]=new InFile(p);
            else
                allTest[i]=null;//superfluous
            String p2=basePath+classifiers[i]+"/"+classifiers[i]+"TrainCVAcc.csv";
            if(new File(p2).exists())
                allTrainCV[i]=new InFile(p2);
            else
                allTrainCV[i]=null;//superfluous
        }
        
        for(String prob:problems){
            testAcc.writeString(prob+",");
            testStdDev.writeString(prob+",");
            trainCVAcc.writeString(prob+",");
            trainCVStdDev.writeString(prob+",");
            diffAcc.writeString(prob+",");
            diffStdDev.writeString(prob+",");
            count.writeString(prob+",");
            String prev="First";
            for(int j=0;j<allTest.length;j++){
                if(allTrainCV[j]==null){
                    trainCVAcc.writeString(",");
                }
                if(allTest[j]==null){
                    testAcc.writeString(",");
                    count.writeString("0,");
                }
                else{//Find mean
                    try{
                        String r=allTest[j].readLine();
                        String[] res=r.split(",");
                        count.writeString((res.length-1)+",");
                        double mean=0;
                        double sumSquare=0;
                        for(int k=1;k<res.length;k++){
                            double d=Double.parseDouble(res[k]);
                            mean+=d;
                            sumSquare+=d*d;
                        }
                        if(res.length>1){
                            int size=(res.length-1);
                            mean=mean/size;
                            double stdDev=sumSquare/size-mean*mean;
                            stdDev=Math.sqrt(stdDev);
                            testAcc.writeString(df.format(mean)+",");
                            testStdDev.writeString(df.format(stdDev)+",");
                        }
                        else{
                            testAcc.writeString(",");
                            testStdDev.writeString(",");
                        }
                        prev=r;
                    if(allTrainCV[j]!=null){ //Find train and diffs
                        String r2=allTrainCV[j].readLine();
                        String[] res2=r2.split(",");
                        double mean2=0;
                        double sumSquare2=0;
                        for(int k=1;k<res2.length;k++){
                            double d2=Double.parseDouble(res2[k]);
                            mean2+=d2;
                            sumSquare2+=d2*d2;
                        }
                        if(res2.length>1){
                            int size=(res2.length-1);
                            mean2=mean2/size;
                            double stdDev2=sumSquare2/size-mean2*mean2;
                            stdDev2=Math.sqrt(stdDev2);
                            trainCVAcc.writeString(df.format(mean2)+",");
                            trainCVStdDev.writeString(df.format(stdDev2)+",");
                            if(res.length==res2.length){//Can do the diff
                                double meanDiff=0;
                                double sumSquareDiff=0;
                                for(int k=1;k<res2.length;k++){
                                    double d1=Double.parseDouble(res[k]);
                                    double d2=Double.parseDouble(res2[k]);
                                    meanDiff+=(d1-d2);
                                    sumSquareDiff+=(d1-d2)*(d1-d2);
                                }
                                meanDiff=meanDiff/res.length;
                                double stdDevDiff=sumSquareDiff/res.length-mean2*mean2;
                                stdDevDiff=Math.sqrt(stdDevDiff);
                                diffAcc.writeString(df.format(meanDiff)+",");
                                diffStdDev.writeString(df.format(stdDevDiff)+",");
                            }
                        }else{
                            diffAcc.writeString(",");
                            diffStdDev.writeString(",");
                        }
                        }
                        else{
                            trainCVAcc.writeString(",");
                            trainCVStdDev.writeString(",");
                        }            
                    }catch(Exception ex){
                        System.out.println("failed to read line: "+ex+" previous line = "+prev);
                    }
                }        
            }
            testAcc.writeString("\n");
            testStdDev.writeString("\n");
            count.writeString("\n");
            trainCVAcc.writeString("\n");
            trainCVStdDev.writeString("\n");
            diffAcc.writeString("\n");
            diffStdDev.writeString("\n");
        }
        for(InFile  inf:allTest)
            if(inf!=null)
                inf.closeFile();
        testAcc.closeFile();
        count.closeFile();
        testStdDev.closeFile();
        trainCVAcc.closeFile();
        trainCVStdDev.closeFile();
        diffAcc.closeFile();
        diffStdDev.closeFile();
             
         
    }
    
    public static void collate(String[] args){
//STAGE 1: Read from arguments, find problems        
        readData(args);
        System.out.println(" number of classifiers ="+numClassifiers);
//STAGE 2: Collate the individual fold files into one        
        collateFolds();
//STAGE 3: Summarise over folds 
        averageOverFolds();
       
        
    }
//First argument: String path to results directories
//Second argument: path to directory with problem names to look for
//Third argument: number of folds    
//Next x arguments: x Classifiers to collate    
//Next x arguments: number of numParas stored for each classifier    
    public static void main(String[] args) {
        if(args.length>1)
            collate(args);
        else{    
            String[] str={"C:/Research/Results/RepoResults/","C:/Users/ajb/Dropbox/TSC Problems/","30","SVM","2"};
            collate(str);
        
        }
    }
    
}
