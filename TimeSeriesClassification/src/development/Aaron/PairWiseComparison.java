/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development.Aaron;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

/**
 *
 * @author raj09hxu
 */
public class PairWiseComparison {
    
    final static String location = "C:/LocalData/Dropbox/Big TSC Bake Off/New Results/shapelet/ST";   
    
    
    public static void main(String[] args){
        
        String partialDir = location + "/" + "Full transform with partial results";
        
        System.out.println("File,Fold,Full,Partial");
        
        File f = new File(partialDir);
        //find all the datasets with partial results.
        for(File partial : f.listFiles()){
            if(!partial.isDirectory()) continue;
            //find each fold and assess accuracy
            
            for(int i=0; i<100; i++){
                File foldFile = new File(partial.getAbsolutePath() + "/" + "fold" +i +".csv");
                if(!foldFile.exists()) continue;
                
                readFoldCSV(foldFile, partial.getName(), i);
            }
        }
    }
    
    private static void readFoldCSV(File foldFile, String dataset, int fold){
        File f = new File(location + "/" + "Predictions" + "/" + dataset + "/" + "fold" + fold + ".csv");
        
        if(!f.exists()) return;
        
        try{
            System.out.println(dataset+","+fold+","+readCSV(foldFile)+","+readCSV(f));
        }catch(IOException ex){
            System.out.println(ex);
        }
    }
    
    private static double readCSV(File foldFile) throws FileNotFoundException{
        Scanner sc = new Scanner(foldFile);
        
        int correct = 0;
        int total =0;
        
        while(sc.hasNextLine()){
            total++;
            
            String line = sc.nextLine();
            String[] values = line.split(",");
            
            if(values[0].equalsIgnoreCase(values[1])){
                correct++;
            }
        }

        return (double) correct / (double) total;
    }
    
}
