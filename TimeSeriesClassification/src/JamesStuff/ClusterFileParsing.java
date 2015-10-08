
package JamesStuff;

import development.DataSets;
import fileIO.*;

public class ClusterFileParsing {
    
    static String[] classifiers = { "BOP", "SAXVSM", "BOSS" };
    static int numDatasets = 85;
    static int numResamples = 100;
    
    static String masterPath = "C:\\JAMESBAKEOFFINTERN\\ClusterFiles\\";
    static String commandPath = masterPath + "Commands\\";
    static String inResultsPath = masterPath + "ClusterResults\\";
    static String outResultsPath = masterPath + "FinalResults\\";
    
    public static int[] completions = { };
    
    public static void genScriptFiles() { 
        OutFile puttyCommands = new OutFile(commandPath + "commands.csv");
        
        BOP_SAXVSMscripts(puttyCommands);
        BOSSscripts(puttyCommands);
        
        puttyCommands.closeFile();
    }
    
    public static void BOP_SAXVSMscripts(OutFile puttyCommands) {
        for (int c = 0; c < classifiers.length - 1; ++c) {
            String classifier = classifiers[c];
            
            for (int dataset = 0; dataset < numDatasets; ++dataset) {
                String exp = classifier+"_"+dataset;
                String filename = exp+".bsub";
                String command = "bsub < " + filename;
                
                puttyCommands.writeLine(command);
                
                OutFile script = new OutFile(commandPath+"\\"+classifier+"\\"+filename);
                script.writeLine("#!/bin/csh");
                script.writeLine("#BSUB -q long");
                script.writeLine("#BSUB -J "+exp+"[1-"+numResamples+"]");
                script.writeLine("#BSUB -oo output/"+exp+"_%I.out");
                script.writeLine("#BSUB -eo error/"+exp+"_%I.err");
                script.writeLine("#BSUB -R \"rusage[mem=4000]\"");
                script.writeLine("#BSUB -M 12000");
                script.writeLine("");
                script.writeLine("module add java/jdk/1.8.0_31");
                script.writeLine("java -jar DictionaryBased3.jar "+exp+" $LSB_JOBINDEX");
                script.closeFile();
            }
        }
    }
    
    public static void BOSSscripts(OutFile puttyCommands) {
        String classifier = classifiers[classifiers.length-1]; //boss
        
        for (int dataset = 0; dataset < numDatasets; ++dataset) {
            int resamplesToDo = numResamples - completions[dataset];
            
            if (resamplesToDo == 0) 
                continue;
            
            String exp = classifier+"_"+dataset;
            String filename = exp+".bsub";
            String command = "bsub < " + filename;

            puttyCommands.writeLine(command);

            OutFile script = new OutFile(commandPath+"\\"+classifier+"\\"+filename);
            script.writeLine("#!/bin/csh");
            script.writeLine("#BSUB -q long");
            script.writeLine("#BSUB -J "+exp+"[1-"+resamplesToDo+"]");
            script.writeLine("#BSUB -oo output/"+exp+"_%I.out");
            script.writeLine("#BSUB -eo error/"+exp+"_%I.err");
            script.writeLine("#BSUB -R \"rusage[mem=4000]\"");
            script.writeLine("#BSUB -M 12000");
            script.writeLine("");
            script.writeLine("module add java/jdk/1.8.0_31");
            script.writeLine("java -jar DictionaryBased3.jar "+exp+" $LSB_JOBINDEX");
            script.closeFile();
        }
    }
    
    public static void collateResultsFiles() {
        for (String classifier : classifiers) {
            for (String dataset : DataSets.fileNames) {
                OutFile out = new OutFile(outResultsPath + "\\" + classifier + "\\" + dataset + ".csv");
                out.writeString(dataset+",");
                
                for (int resample = 0; resample < numResamples; ++resample) {
                    InFile in = new InFile(inResultsPath + "\\" + classifier + "\\" + dataset + "\\" + resample);
                    out.writeString(in.readDouble()+",");
                    in.closeFile();
                }
                
                out.writeString("\n");
                out.closeFile();
            }
        }
    }
    
    public static void findBossCompletions() { 
        
        //run this and copy results into static array in clusterFileParsing
        
        int[] completed = new int[numDatasets];
        
        String bossResultsLoc = "C:\\Temp\\BAKEOFFRESULTS\\BOSS\\";
        
        for (int i = 0; i < numDatasets; ++i) {
            InFile in = new InFile(DataSets.fileNames[i]+".csv", ',');
            
            completed[i] = -1; //ignore dataset name at start
            
            try {
                while(true) {
                    in.readDouble();
                    ++completed[i];
                }
            } catch (Exception e) { }
            
            in.closeFile();
        } 
        
        System.out.print(completed[0]);
        for (int i = 0; i < numDatasets; ++i) {
            System.out.print(","+completed[i]);
        }
    }
    
    public static void main(String[] args) throws Exception {
         findBossCompletions();
         
//         genScriptFiles(); //once filled in boss completions, uncomment and run
    }
}






