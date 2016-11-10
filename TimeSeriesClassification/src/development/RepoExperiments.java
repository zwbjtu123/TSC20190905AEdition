/**
 *
 * @author ajb
 *local class to run experiments with the repo data
* this class can do all the messy stuff, generating scripts etc, then
* just call the methods in Bagnall16bakeoff.java

 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import static papers.Bagnall16bakeoff.singleClassifierAndFold;

public class RepoExperiments {

    static String[] classifiers={"HESCA","RISE"};

    
    public static void createBaseExperimentScripts(boolean grace){
//Generates cluster scripts for all combos of classifier and data set
//Generates txt files to run jobs for a single classifier        
        String path="C:\\Users\\ajb\\Dropbox\\Code\\Cluster Scripts\\RepoScripts\\";
        File f=new File(path);
        int folds=100; 
        int mem=8000;
        if(!f.isDirectory())
            f.mkdir();
        for(String s:classifiers){
            OutFile of2;
            if(grace)
                of2=new OutFile(path+s+"Grace.txt");
            else
                of2=new OutFile(path+s+".txt");
            for(String a:DataSets.fileNames){
                OutFile of;
                if(grace)
                    of = new OutFile(path+s+a+"Grace.bsub");
                else
                    of = new OutFile(path+s+a+".bsub");
                of.writeLine("#!/bin/csh");
                if(grace)
                    of.writeLine("#BSUB -q short");
                else
                    of.writeLine("#BSUB -q long-eth");
                of.writeLine("#BSUB -J "+s+a+"[1-"+folds+"]");
                of.writeLine("#BSUB -oo output/"+a+".out");
                of.writeLine("#BSUB -eo error/"+a+".err");
                if(grace){
                    of.writeLine("#BSUB -R \"rusage[mem="+mem+"]\"");
                    of.writeLine("#BSUB -M "+mem);
                    of.writeLine(" module add java/jdk/1.8.0_31");
                }
                else{
                    of.writeLine("#BSUB -R \"rusage[mem="+(4000+mem)+"]\"");
                    of.writeLine("#BSUB -M "+(4000+mem));
                    of.writeLine("module add java/jdk1.8.0_51");
                }
                of.writeLine("java -jar Repo.jar "+s+" "+a+" $LSB_JOBINDEX");                
                if(grace)
                    of2.writeLine("bsub < Scripts/RepoScripts/"+s+a+"Grace.bsub");
                else
                    of2.writeLine("bsub < Scripts/RepoScripts/"+s+a+".bsub");
            }   
        }
    } 
    public static boolean foldComplete(String path, int fold, int numTrain,int numTest){
       boolean complete=true;
//Check both train and test present
      File f=new File(path+"//testFold"+fold+".csv");
      File f2=new File(path+"//trainFold"+fold+".csv");
      if(!f.exists()||!f2.exists())//Neither exist
          return false;
      else{
          InFile inf1=new InFile(path+"//testFold"+fold+".csv");
          InFile inf2=new InFile(path+"//testFold"+fold+".csv");
//Check number of lines
          int c1=inf1.countLines();
          int c2=inf2.countLines();
          if(c1!=(numTest+3) || c2!=(numTrain+3))
              return false;
      }
      return true;
    }
    public static void collateResults(int folds){
        String basePath="C:\\Research\\Results\\RepoResults\\";
//1. Collate single folds into single classifier_problem files        
        for(String cls:classifiers){
//Check classifier directory exists. 
            File f=new File(basePath+cls);
            if(f.isDirectory()){
//Write collated results for this classifier to a single file                
                OutFile clsResults=new OutFile(basePath+cls+"//"+cls+".csv");
                OutFile missing=null;
                for(int i=0;i<DataSets.fileNames.length;i++){
                    clsResults.writeString(DataSets.fileNames[i]+",");
                    String path=basePath+cls+"//Predictions//"+DataSets.fileNames[i];
                    for(int j=0;j<folds;j++){
    //Check fold exists
                        if(foldComplete(path,j,1,1)){ //This could fail if file partial//
    //Read in accuracy and store                    
                            InFile inf=new InFile(path+"//testFold"+j+".csv");
                            inf.readLine();
                            inf.readLine();
                            clsResults.writeString(inf.readDouble()+",");    
                        }
                        else{
                            if(missing==null)
                                missing=new OutFile(basePath+cls+"//"+cls+"MISSING.csv");
                            missing.writeLine("fold,"+j);
                        }
                    }
                    clsResults.writeString("\n");
                }
                clsResults.closeFile();
            }
        }
//3. Merge classifier files into a single file with average accuracies
        OutFile acc=new OutFile(basePath+"CombinedAcc.csv");
        OutFile count=new OutFile(basePath+"CombinedCount.csv");
        for(String cls:classifiers){
            acc.writeString(","+cls);
            count.writeString(","+cls);
        }
        acc.writeString("\n");
        count.writeString("\n");
        InFile[] all=new InFile[classifiers.length];
        for(int i=0;i<all.length;i++){
            String p=basePath+classifiers[i]+"//"+classifiers[i]+".csv";
            if(new File(p).exists())
                all[i]=new InFile(p);
            else
                all[i]=null;//superfluous
        }
        for(int i=0;i<DataSets.fileNames.length;i++){
            acc.writeString(DataSets.fileNames[i]+",");
            count.writeString(DataSets.fileNames[i]+",");
            for(int j=0;j<all.length;j++){
                if(all[j]==null){
                    acc.writeString(",");
                    count.writeString("0,");
                }
                else{//Find mean
                    String[] res=all[j].readLine().split(",");
                    count.writeString((res.length-1)+",");
                    double mean=0;
                    for(int k=1;k<res.length;k++)
                        mean+=Double.parseDouble(res[k]);
                    if(res.length>1)
                        acc.writeString((mean/(res.length-1))+",");
                    else
                        acc.writeString(",");
                }
            } 
            acc.writeString("\n");
            count.writeString("\n");
        }
        
    }
    public static void main(String[] args){
        collateResults(100);
        createBaseExperimentScripts(true);
       createBaseExperimentScripts(false);
        System.exit(0);
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/RepoExperiments/";
            singleClassifierAndFold(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/RepoExperiments/";
            String[] paras={"RISE","ItalyPowerDemand","6"};
            
            singleClassifierAndFold(paras);            
        }
    }
}
