/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development.Aaron;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 *
 * @author Aaron
 */
public class ProcessSpeedAndAccuracyFiles {
    
    public static void main(String[] args) throws IOException{
        String filePath = "C:/LocalData/Experiments";
        File dir = new File(filePath);

        File output = new File(filePath + "/accuracy.csv");
        output.createNewFile();
        PrintWriter outputAccuracy = new PrintWriter(output);
        
        output = new File(filePath + "/counts.csv");
        output.createNewFile();
        PrintWriter outputCounts = new PrintWriter(output);
        
        
        for(File f : dir.listFiles()){
            if(f.isDirectory())
                processFiles(outputCounts, outputAccuracy, f);
        } 
        
        outputAccuracy.close();
        outputCounts.close();
    }
    
    private static void processFiles(PrintWriter counts, PrintWriter accuracy, File dir) throws FileNotFoundException{
        //f is the directory containing our 7 files.
        String dataset = dir.getName();
       
        counts.printf("%s",dataset);
        accuracy.printf("%s",dataset);
        Scanner sc;
        
        String[] count = new String[7];
        String[] acc = new String[7];
        
        System.out.println(dir.getName());
        
        for(File f : dir.listFiles()){
            sc = new Scanner(f);
            
            String line = sc.nextLine();
            String[] data = line.split(",");
            
            //this ensures correct ordering of values.
            SpeedAndAccuracyExperiments.Parameters param = SpeedAndAccuracyExperiments.Parameters.valueOf(data[0]);
            count[param.ordinal()] = data[1];
            acc[param.ordinal()] = data[2];
        }
        
        
        for(String s : count){
            counts.printf(",%s", s);
        }
        
         for(String s : acc){
            accuracy.printf(",%s", s);
        }
        
        counts.printf("\n");
        accuracy.printf("\n");
        
    }
    
}
