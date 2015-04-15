/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package old_development;

import fileIO.InFile;
import fileIO.OutFile;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author ajb
 */
public  class RandTest extends Thread{
    String fileName;
    public RandTest(String s){
        fileName=s;
    }
    public void run(){
        OutFile of =new OutFile(fileName);
        for(int i=0;i<1000;i++){
            Random r = new Random(i);
            for(int j=0;j<1000;j++){
                of.writeString(r.nextInt()+",");
            }
            of.writeString("\n");
        }
    }
    public static void compareRands(String s1, String s2){
        InFile f1= new InFile(s1);
        InFile f2= new InFile(s2);
        for(int i=0;i<1000;i++)
            for(int j=0;j<1000;j++){
                int a = f1.readInt();
                int b=f2.readInt();
                if(a!=b){
                    System.out.println("RANDS NOT EQUAL : Seed ="+i+" order ="+j+" values ="+a+" and "+b);
                    return;
                }
            }
               
        
    }
    public static void main(String[] args){
        RandTest r1=new RandTest("randTest1.csv"), r2=new RandTest("randTest2.csv");
        
       try {
        r1.start();
        r2.start();
        r1.join();
        r2.join();
        compareRands("randTest1.csv","randTest2.csv");
    } catch (InterruptedException ex) {
                    System.out.println(" ERROR in RandTest: "+ex);
    }

        compareRands("randTest1.csv","randTest2.csv");
    }
}