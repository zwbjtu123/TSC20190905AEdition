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
public class ResultsCollation {
    static String[] elastic = {"Euclidean_1NN","DTW_R1_1NN","DTW_Rn_1NN","DDTW_R1_1NN","DDTW_Rn_1NN","ERP_1NN","LCSS_1NN","MSM_1NN","TWE_1NN","WDDTW_1NN","WDTW_1NN","DD_DTW","DTD_C"};
    static String[] standard={"NB","C45","SVML","SVMQ","Logistic","BayesNet","RandF","RotF","MLP"};
    static String[] dictionary={"BoP","SAXVSM","BOSS"};
    static String[] shapelet={"ST","LS","FS"};
    static String[] interval={"TSF","TSBF","LPS"};
    static String[] complexity={"CID_EE","CID_DTW","RPCD"};
    static String[] ensemble={"EE","COTE"};
    static final String[][] names={standard,elastic,shapelet,dictionary,interval,ensemble};
    static final String[] directoryNames={"standard","Elastic distance measures","shapelet","dictionary","interval","ensemble"};

    /*names[0]=standard;
        names[1]=elastic;
        names[2]=shapelet;
        names[3]=dictionary;
        names[4]=interval;
        names[5]=ensemble;   
    */
    
/** 
 * 2. single file of problem and classifier  to single file of classifier
 * NaiveBayes/Adiac.csv
 * NaiveBayes/ArrowHead.csv    
 * */
    public static void clusterResultsCollation(String path){
        for(int i=0;i<directoryNames.length;i++){
            for(int j=0;j<names[i].length;j++){
//Check for directory
                File dir= new File(path+"\\"+directoryNames[i]+"\\"+names[i][j]);
                if(dir.isDirectory()){    //Proceed if there is a directory of results
                    System.out.println(path+"\\"+directoryNames[i]+"\\"+names[i][j]+" IS a directory");
                    OutFile results = new OutFile(path+"\\"+directoryNames[i]+"\\"+names[i][j]+".csv");
                    for(String s:DataSets.fileNames){
                        File f= new File(path+"\\"+directoryNames[i]+"\\"+names[i][j]+"\\"+s+".csv");
                        if(f.exists()){
                            InFile f2=new InFile(path+"\\"+directoryNames[i]+"\\"+names[i][j]+"\\"+s+".csv"); 
                            results.writeLine(f2.readLine());
                        }
                    }
                }
                else{
                   System.out.println(path+"\\"+directoryNames[i]+"\\"+names[i][j]+" IS NOT a directory");

               }
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
            r.mean=-1;
            r.name=split[0];
            if(split.length>1){
                r.accs=new double[split.length-1];
    //            System.out.println("length="+r.accs.length+"::::"+line);
                for(int j=0;j<r.accs.length;j++){
                    try{
                        r.accs[j]=Double.parseDouble(split[j+1]);
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

        
        for(int i=0;i<directoryNames.length;i++){
            for(int j=0;j<names[i].length;j++){        
                for(String s:names[i]){
                    File f= new File(path+"\\"+directoryNames[i]+"\\"+s+".csv");
                    if(f.exists()){
                        InFile inf=new InFile(path+"\\"+directoryNames[i]+"\\"+s+".csv");
                        OutFile outf=new OutFile(path+"\\SingleClassifiers\\"+s+".csv");
                        incompleteResultsParser(inf,outf);
                    }
                    else
                        System.out.println(" File "+path+"\\"+directoryNames[i]+"\\"+s+" does not exist");
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
        
    public static void main(String[] args)throws Exception{
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
        System.exit(0);
  */      
        String root ="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results";
        clusterResultsCollation(root);
        fileStandardiseForProblems(root);
        fileCombineClassifiers(root+"\\SingleClassifiers\\",root);
    }

}
