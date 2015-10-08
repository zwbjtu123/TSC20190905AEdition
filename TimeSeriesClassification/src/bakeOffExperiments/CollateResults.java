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
package bakeOffExperiments;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;

/**
 *
 * @author ajb
 */
public class CollateResults {
   static final String[][] names={Experiments.standard,Experiments.elastic,Experiments.shapelet,Experiments.dictionary,Experiments.interval,Experiments.ensemble,Experiments.complexity};
   static String[] dirNames=Experiments.directoryNames;
    
/** 
 * 2. single file of problem and classifier  to single file of classifier
 * NaiveBayes/Adiac.csv
 * NaiveBayes/ArrowHead.csv    
 * */
    public static void clusterResultsCollation(String path){
        for(int i=0;i<dirNames.length;i++){
            for(int j=0;j<names[i].length;j++){
//Check for directory
                File dir= new File(path+"\\"+dirNames[i]+"\\"+names[i][j]);
                if(dir.isDirectory()){    //Proceed if there is a directory of results
//                  System.out.println(path+"\\"+dirNames[i]+"\\"+names[i][j]+" IS a directory");
                    OutFile results = new OutFile(path+"\\"+dirNames[i]+"\\"+names[i][j]+".csv");
                    for(String s:DataSets.fileNames){
                        File f= new File(path+"\\"+dirNames[i]+"\\"+names[i][j]+"\\"+s+".csv");
                        if(f.exists()){
                            InFile f2=new InFile(path+"\\"+dirNames[i]+"\\"+names[i][j]+"\\"+s+".csv"); 
                            results.writeLine(f2.readLine());
                        }
                    }
                }
 //               else{
   //                System.out.println(path+"\\"+dirNames[i]+"\\"+names[i][j]+" IS NOT a directory");
//
  //             }
            }
        }
    }
 
/* Takes a possibly partial list of results and format into the outfile */
    public static class Results{
        String name;
        double mean;
        double stdDev;
        double[] accs;
        public boolean equals(Object o){
            if( ((Results)o).name.equals(this.name))
                return true;
            return false;
        }
    }
//OUTPUT TO C:\Users\ajb\Dropbox\Big TSC Bake Off\New Results\SingleClassifiers    
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
            if(split.length>1){
                r.accs=new double[split.length-1];
    //            System.out.println("length="+r.accs.length+"::::"+line);
                for(int j=0;j<r.accs.length;j++){
                    try{
                        r.accs[j]=Double.parseDouble(split[j+1]);
                        if(r.accs[j]>1)
                           r.accs[j]=r.accs[j-1]; //REMOVE IMMEDIATELY
//                            throw new Exception("ERRPR ACCURACY >1 = "+r.accs[j]+" line ="+i+" split ="+j+" file = "+f.getName());
                        r.mean+=r.accs[j];
                    }catch(Exception e){
                        System.out.println("ERROR: "+split[j]+" giving error "+e+" in file "+f.getName());
                        System.exit(0);
                    }
                }
                r.mean/=r.accs.length;
                r.stdDev=0;
                for(int j=0;j<r.accs.length;j++){
                    r.stdDev+=(r.accs[j]-r.mean)*(r.accs[j]-r.mean);
                }
                r.stdDev/=(r.accs.length-1);
                r.stdDev=Math.sqrt(r.stdDev);
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
                    of.writeLine(r.mean+","+r.stdDev+","+r.accs.length);
                else
                    of.writeLine("");
            }
            else
                of.writeLine("");
        }
    }    
    

    public static void fileStandardiseForProblems(String path) throws Exception{

        
        for(int i=0;i<dirNames.length;i++){
            for(int j=0;j<names[i].length;j++){        
                for(String s:names[i]){
                    File f= new File(path+"\\"+dirNames[i]+"\\"+s+".csv");
                    if(f.exists()){
                        InFile inf=new InFile(path+"\\"+dirNames[i]+"\\"+s+".csv");
                        OutFile outf=new OutFile(path+"\\SingleClassifiers\\"+s+".csv");
                        incompleteResultsParser(inf,outf);
                    }
                    else
                        System.out.println(" File "+path+"\\"+dirNames[i]+"\\"+s+" does not exist");
                }
            }
        }
    }
    public static void fileCombineClassifiers(String inPath,String outPath) throws Exception{
        OutFile[] of={new OutFile(outPath+"\\Means.csv"),new OutFile(outPath+"\\StDevs.csv"),new OutFile(outPath+"\\SampleSizes.csv")};

        
        for(OutFile o:of){
            o.writeString(",");
            for(String[] n:names)
            for(String st:n)
                o.writeString(st+",");
        }
        for(OutFile o:of)
            o.writeString("\n");
        
//Try open all
        InFile[][] inFiles=new InFile[names.length][];
        for(int i=0;i<inFiles.length;i++){
            inFiles[i]=new InFile[names[i].length];
            for(int j=0;j<inFiles[i].length;j++){
//Check existance
                File f=new File(inPath+names[i][j]+".csv");
//If exists, open                
                if(f.exists())
                    inFiles[i][j]=new InFile(inPath+names[i][j]+".csv");
                else{
                    inFiles[i][j]=null;
//                    System.out.println(" File "+names[i][j]+" does not exist");
                }
            }
        }
        for(String s:DataSets.fileNames){
            for(OutFile o:of)
                o.writeString(s+",");
            for(int i=0;i<inFiles.length;i++){
                for(int j=0;j<inFiles[i].length;j++){
                    if(inFiles[i][j]==null){
                        for(OutFile o:of)
                            o.writeString(",");
                    }
                    else{
                        String[] name=inFiles[i][j].readLine().split(",");
                        if(name.length<2){
                            for(OutFile o:of)
                                o.writeString(",");
                        }
                        else{
                            for(int k=0;k<3;k++)
                                of[k].writeString(name[k+1]+",");
                        }
                    }
                }
            }    
            for(OutFile o:of)
               o.writeString("\n");
        }
    }
    public static void parseSingleProblem(String path,String result, String problem){
//Check they all exist        
        for(int i=0;i<100;i++){
            File f= new File(path+"\\fold"+i+".csv");
            InFile inf= new InFile(path+"\\fold"+i+".csv");
            int cases=inf.countLines();
            if(!f.exists() || cases==0){
                System.out.println(" Incomplete files, no fold "+i+" on path ="+path);
                System.exit(0);
            }
            else{
                
            }
        }
        OutFile out=new OutFile(result);
        out.writeString(problem+",");
        for(int i=0;i<100;i++){
            InFile inf= new InFile(path+"\\fold"+i+".csv");
            int cases=inf.countLines();
            inf= new InFile(path+"\\fold"+i+".csv");
            double acc=0;
            for(int j=0;j<cases;j++){
                int act=inf.readInt();
                int pred=inf.readInt();
                if(act==pred){
                    acc++;
                }
            }
            acc/=cases;
            out.writeString(acc+",");
            System.out.println("Fold "+i+" acc ="+acc);
        }
    }
        
    public static void main(String[] args)throws Exception{
        String r="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\";
        String problem="FordA";
        String classifier="RotF";
        String result=r+"standard\\"+classifier+"\\"+problem+".csv";
        String source=r+"standard\\"+classifier+"\\Predictions\\"+problem;
        parseSingleProblem(source,result,problem);
        problem="FordB";
        classifier="RotF";
        result=r+"standard\\"+classifier+"\\"+problem+".csv";
        source=r+"standard\\"+classifier+"\\Predictions\\"+problem;
        parseSingleProblem(source,result,problem);
        System.exit(0);
 /*       String r="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\SingleClassifiers\\MLP.csv";
       InFile in= new InFile(r);
       int l=in.countLines();
       in.reopen();
       for(int i=0;i<l;i++){
            String[] name=in.readLine().split(",");
            System.out.print(" Line "+i+" splits = "+name.length+" ");
            for(String s:name) 
                System.out.print(s+ " ");
            System.out.print("\n");
       }
  */      
        String root ="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results";
        clusterResultsCollation(root);
        fileStandardiseForProblems(root);
        fileCombineClassifiers(root+"\\SingleClassifiers\\",root);
    }

}
