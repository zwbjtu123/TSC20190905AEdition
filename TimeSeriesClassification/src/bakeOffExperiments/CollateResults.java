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
                    OutFile results = new OutFile(path+"\\"+dirNames[i]+"\\"+names[i][j]+".csv");
                    for(String s:DataSets.fileNames){
                        File f= new File(path+"\\"+dirNames[i]+"\\"+names[i][j]+"\\"+s+".csv");
                        if(f.exists()){
                            InFile f2=new InFile(path+"\\"+dirNames[i]+"\\"+names[i][j]+"\\"+s+".csv");
                            String str=f2.readLine();
                            if(names[i][j].equals("DTD_C") && s.equals("InsectWingbeatSound"))  
                            System.out.println(path+"\\"+dirNames[i]+"\\"+names[i][j]+" IS a directory: file f content ="+str);
                            results.writeLine(str);
                        }
                    }
                    results.closeFile();
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
            for(String s:names[i]){
                File f= new File(path+"\\"+dirNames[i]+"\\"+s+".csv");
                if(f.exists()){
                    InFile inf=new InFile(path+"\\"+dirNames[i]+"\\"+s+".csv");
                    OutFile outf=new OutFile(path+"\\SingleClassifiers\\"+s+".csv");
                    incompleteResultsParser(inf,outf);
                    inf.closeFile();
                    outf.closeFile();
                }
                else
                    System.out.println(" File "+path+"\\"+dirNames[i]+"\\"+s+" does not exist");
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
                            of[0].writeString(",");
                            of[1].writeString(",");
                            of[2].writeString("0,");
                    }
                    else{
                        String[] name=inFiles[i][j].readLine().split(",");
                        if(name.length<2){
                            of[0].writeString(",");
                            of[1].writeString(",");
                            of[2].writeString("0,");
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
            inf.closeFile();
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
            inf.closeFile();
        }
    }
     public static void combineFolds(String source, OutFile dest, int start, int end){
        for(int i=start;i<=end;i++){
            File inf=new File(source+"fold"+i+".csv");
            if(inf.exists() && inf.length()>0){
                InFile f=new InFile(source+"fold"+i+".csv");
                int lines=f.countLines();
                f=new InFile(source+"fold"+i+".csv");
                double acc=0;
                for(int j=0;j<lines;j++){
                    double act=f.readDouble();
                    double pred=f.readDouble();
                    if(act==pred) acc++;
                }
                if(i<end)
                    dest.writeString(acc/lines+",");
                else
                    dest.writeLine(acc/lines+"");
            }
        }
    }

    public static void combineSingles(String root){
        
        for(int i=0;i<dirNames.length;i++){
            for(int j=0;j<names[i].length;j++){
//Check for directory
                File dir= new File(root+"\\"+dirNames[i]+"\\"+names[i][j]);
                if(dir.isDirectory()){    //Proceed if there is a directory of results
    //Check if already complete
                        if(names.equals("DTD_C")){
                            System.out.println(" Processing DTD_C");    
                        }
                    for(int k=0;k<DataSets.fileNames.length;k++){
                        String cls=names[i][j];
                        String directory=dirNames[i];
                        String s= DataSets.fileNames[k];
                        File f=new File(dir+"\\"+s+".csv");
                        if(f.exists()){ //At least partially complete
    //See how complete it is
                            InFile inf=new InFile(dir+"\\"+s+".csv");
                            String line=inf.readLine();
                            inf.closeFile();
//                            System.out.println(directory+" "+cls+" "+s+" "+line);
                            String[] res=null;
                            if(line==null){ //Delete the file
                                f=new File(dir+"\\"+s+".csv");
                                f.delete();
                            }
                            else{
                                res=line.split(",");
                            }
                            if(res!=null && res.length<101)
                            {   //Check to see if there are any preds need adding
                                OutFile of = new OutFile(dir+"\\"+s+".csv");
                                int length=res.length-1;
                                for(String str:res){
                                    if(str.equals("NaN"))
                                        length--;
                                    else
                                        of.writeString(str+",");
                                }
                                combineFolds(dir+"\\Predictions\\"+s+"\\",of,length,99);
                                of.closeFile();
                                
                            }
                        }
                        else{
                            //Check if there are any predictions
                            if(checkPredictions(dir+"\\Predictions\\"+s+"\\")){
                                OutFile of=new OutFile(dir+"\\"+s+".csv"); 
                                of.writeString(s+",");
                                combineFolds(dir+"\\Predictions\\"+s+"\\",of,0,99);
                                of.closeFile();
                            }
                        }
                    }
                }
            }
    /*        prob="MiddlePhalanxOutlineCorrect";
        of=new OutFile(root+"\\"+dir+"\\"+cls+"\\"+prob+"2.csv"); 
        combineFolds(root+"\\"+dir+"\\"+cls+"\\Predictions\\"+prob+"\\",of,27,99);
*/
        }
    }
    public static boolean checkPredictions(String path){
        for(int i=0;i<100;i++){
            if(new File(path+"fold"+i+".csv").exists())
                return true;
        }
        return false;
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
            of.writeLine("java -jar TimeSeriesClassification.jar "+classifier+" " +problem+" $LSB_JOBINDEX"+j);
            of2.writeLine("bsub < "+"Scripts/"+classifier+"/Unstarted/"+problem+".bsub");
            of.closeFile();
        }
        
    }
    public static int getParas(String algo){
        switch(algo){
            case "TSBF": return 4;
            case "LS": return 8;
            default: return 10;    
        }
    }
    public static void generateScripts(String root, String dest){
        InFile inf=new InFile(root+"\\SampleSizes.csv");
        OutFile outf=new OutFile(dest+"\\AllProblems.txt");
        OutFile outf2=new OutFile(dest+"\\UnstartedProblems.txt");
        File f=new File(dest+"\\Scripts");
        deleteDirectory(f);
        if(!f.isDirectory())
            f.mkdir();
        String[] algos=new String[39];
        for(int i=0;i<algos.length;i++)
           algos[i]=inf.readString();
        String[] problems=new String[85];
        int[][] counts=new int[85][algos.length];
        int c=0;
        int p=0;
        for(int i=0;i<problems.length;i++){
            problems[i]=inf.readString();
            for(int j=0;j<algos.length;j++){
                   counts[i][j]=inf.readInt();
//                System.out.print(counts[i][j]+" ");
            }
        }
        for(int j=0;j<algos.length;j++){
            if(generateScripts(algos[j])){
                for(int i=0;i<problems.length;i++){
                    if(counts[i][j]==0){
                        f=new File(dest+"\\Scripts\\"+algos[j]);
                        if(!f.isDirectory())
                            f.mkdir();
                        f=new File(dest+"\\Scripts\\"+algos[j]+"\\Unstarted");
                        if(!f.isDirectory())
                            f.mkdir();
                        c++;
                        p+=100;
                        int paras=getParas(algos[j]);
                        for(int k=1;k<=paras;k++){
                            OutFile of=new OutFile(dest+"\\Scripts\\"+algos[j]+"\\Unstarted\\"+problems[i]+"_"+k+".bsub");
                             of.writeString("#!/bin/csh\n" +
                                "#BSUB -q ");
                             of.writeString("long-ib\n#BSUB -J ");
                            of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-10]");
                            of.writeString("#BSUB -oo output/"+algos[j]+k+"%I.out\n" +
                                "#BSUB -eo error/"+algos[j]+k+"%I.err\n" +
                                "#BSUB -R \"rusage[mem=10000]\"\n" +
                                "#BSUB -M 11000");
                            of.writeLine("\n\n module add java/jdk1.8.0_51");

                            of.writeLine("java -jar TimeSeriesClassification.jar "+algos[j]+" " +problems[i]+" $LSB_JOBINDEX "+k);
                            outf2.writeLine("bsub < "+"Scripts/"+algos[j]+"/Unstarted/"+problems[i]+"_"+k+".bsub");
                            of.closeFile();
                        }
                    }
                    else if(counts[i][j]<100){                    
                        f=new File(dest+"\\Scripts\\"+algos[j]);
                        if(!f.isDirectory())
                            f.mkdir();
                        c++;
                        p+=(100-counts[i][j]);
                        if(algos[j].equals("LS")){
                            int paras=getParas(algos[j]);
                            for(int k=1;k<=paras;k++){
                                OutFile of=new OutFile(dest+"\\Scripts\\"+algos[j]+"\\"+problems[i]+"_"+k+".bsub");
                                 of.writeString("#!/bin/csh\n" +
                                  "#BSUB -q ");
                                 of.writeString("long-ib\n#BSUB -J ");
                                 if(counts[i][j]<=10)
                                    of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-10]");
                                 else
                                    of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-100]");
                                of.writeString("#BSUB -oo output/"+algos[j]+k+"%I.out\n" +
                              "#BSUB -eo error/"+algos[j]+k+"%I.err\n" +
                              "#BSUB -R \"rusage[mem=10000]\"\n" +
                              "#BSUB -M 11000");
                                of.writeLine("\n\n module add java/jdk1.8.0_51");

                                of.writeLine("java -jar TimeSeriesClassification.jar "+algos[j]+" " +problems[i]+" $LSB_JOBINDEX "+k);
                                outf2.writeLine("bsub < "+"Scripts/"+algos[j]+"/Unstarted/"+problems[i]+"_"+k+".bsub");
                                of.closeFile();
                            }                        
                        }
                        else{
                            OutFile of=new OutFile(dest+"\\Scripts\\"+algos[j]+"\\"+problems[i]+".bsub");
                            of.writeString("#!/bin/csh\n" +
                              "#BSUB -q ");
                            if(counts[i][j]>0 && counts[i][j]<9){
                                of.writeString("long-ib\n#BSUB -J ");
                                of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-10]");
                                of.writeString("#BSUB -oo output/"+algos[j]+"%I.out\n" +
                              "#BSUB -eo error/"+algos[j]+"%I.err\n" +
                              "#BSUB -R \"rusage[mem=10000]\"\n" +
                              "#BSUB -M 11000");
                            }else{
                                of.writeString("long\n#BSUB -J ");                           
                                of.writeLine(algos[j]+problems[i]+"["+(counts[i][j]+1)+"-10]");
                                of.writeString("#BSUB -oo output/"+algos[j]+"%I.out\n" +
                                  "#BSUB -eo error/"+algos[j]+"%I.err\n" +
                                  "#BSUB -R \"rusage[mem=2000]\"\n" +
                                  "#BSUB -M 3000");
                            }                            
                            of.writeLine("\n\n module add java/jdk1.8.0_51");
                            of.writeLine("java -jar TimeSeriesClassification.jar "+algos[j]+" " +problems[i]+" $LSB_JOBINDEX");
                            outf.writeLine("bsub < "+"Scripts/"+algos[j]+"/"+problems[i]+".bsub");
                            of.closeFile();
                        }
                    }
                }
            }
        }
        System.out.println(" Total number of problems remaining ="+c);
        System.out.println(" Total number of runs remaining ="+p);
        outf.closeFile();
    }
    public static boolean generateScripts(String algo){
        switch(algo){
            case "TSF":case "TSBF": case "Logistic": case "MLP": case "LS": 
            case "FS": case "ACF": case "PS": 
                return true;
            default:
                return false;
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
    public static void main(String[] args)throws Exception{
        String root="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results";
        combineSingles(root);
        clusterResultsCollation(root);
        fileStandardiseForProblems(root);
        fileCombineClassifiers(root+"\\SingleClassifiers\\",root);
        generateScripts(root,root);

    }
}
