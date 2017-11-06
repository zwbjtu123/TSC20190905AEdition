/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package vector_classifiers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author cjr13geu
 */    
public class ChooseClassifierFromFile implements Classifier{
    
    private Random randomNumber;
    private final int bufferSize = 100000;
    private int foldNumber = 0;
    private int indexOfLargest = 0;
    ArrayList<String> line;
    private String resultsPath = "Results/";
    private String name = "EnsembleResults";  
    private String classifiers[] = {"TunedSVMRBF", "TunedSVMPolynomial"};
    private String relationName = "abalone";
    private double accuracies[];
    private File dir;
    private BufferedReader[] trainFiles;
    private BufferedReader testFile;
    private BufferedWriter outTrain;
    private BufferedWriter outTest;
    
    
    
    public void setFold(int foldNumber){
        this.foldNumber = foldNumber;
    }
    
    public void setClassifiers(String[] classifiers){
        this.classifiers = classifiers;
    }
    
    public void setResultsPath(String resultsPath){
        this.resultsPath = resultsPath;
    }
    
    public void setName(String name){
        this.name = name;
    }
    
    public void setRelationName(String name){
        this.relationName = name;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
 
        dir = new File(resultsPath + "/" + this.name + "/Predictions/" + relationName + "/trainFold" + foldNumber + ".csv");
        
        if(!dir.exists()){
            try{ 
                trainFiles = new BufferedReader[classifiers.length];
                accuracies = new double[classifiers.length];

                for (int i = 0; i < classifiers.length; i++) {
                trainFiles[i] = new BufferedReader(new FileReader(resultsPath + "/"+ classifiers[i] + "/Predictions/" + relationName + "/trainFold" + foldNumber + ".csv"), bufferSize);
                trainFiles[i].mark(bufferSize);
                trainFiles[i].readLine();
                trainFiles[i].readLine();
                accuracies[i] = Double.valueOf(trainFiles[i].readLine());
                }

                for (int i = 0; i < accuracies.length; i++ ) { 
                    if ( accuracies[i] > accuracies[indexOfLargest] ) { 
                        indexOfLargest = i; 
                    }
                }

                ArrayList<Integer> duplicates = new ArrayList<>();
                for (int i = 0; i < accuracies.length; i++) {
                    if(accuracies[indexOfLargest] == accuracies[i] && indexOfLargest != i){
                        duplicates.add(i);
                    }
                }

                randomNumber = new Random(foldNumber);
                if(!duplicates.isEmpty()){
                    indexOfLargest = randomNumber.nextInt(duplicates.size());
                }

                //Write Train file.
                dir = new File(resultsPath + "/" + this.name + "/Predictions/" + relationName);
                dir.mkdirs();
                outTrain = new BufferedWriter(new FileWriter(dir + "/trainFold" + foldNumber + ".csv"));
                trainFiles[indexOfLargest].reset();
                line = new ArrayList<>(Arrays.asList(trainFiles[indexOfLargest].readLine().split(",")));
                line.set(1, name);
                outTrain.write(line.toString().replace("[", "").replace("]", ""));
                outTrain.newLine();

                line = new ArrayList<>(Arrays.asList(trainFiles[indexOfLargest].readLine().split(",")));
                line.add("originalClassifier");
                line.add(classifiers[indexOfLargest]);
                outTrain.write(line.toString().replace("[", "").replace("]", ""));
                outTrain.newLine();

                while((line = new ArrayList<>(Arrays.asList(new String[] { trainFiles[indexOfLargest].readLine() }))).get(0)  != null){
                    outTrain.write(line.get(0));
                    outTrain.newLine();
                }

                //Write Test file.
                outTest = new BufferedWriter(new FileWriter(dir + "/testFold" + foldNumber + ".csv"));
                testFile = new BufferedReader(new FileReader(resultsPath + "/"+ classifiers[indexOfLargest] + "/Predictions/" + relationName + "/testFold" + foldNumber + ".csv"), bufferSize);
                line = new ArrayList<>(Arrays.asList(testFile.readLine().split(",")));
                line.set(1, name);
                outTest.write(line.toString().replace("[", "").replace("]", ""));
                outTest.newLine();

                line = new ArrayList<>(Arrays.asList(testFile.readLine().split(",")));
                line.add("originalClassifier");
                line.add(classifiers[indexOfLargest]);
                outTest.write(line.toString().replace("[", "").replace("]", ""));
                outTest.newLine();

                while((line = new ArrayList<>(Arrays.asList(new String[] { testFile.readLine() }))).get(0)  != null){
                    outTest.write(line.get(0));
                    outTest.newLine();
                }


                for (int i = 0; i < classifiers.length; i++) {
                    trainFiles[i].close();
                    testFile.close();
                }
                outTrain.flush();
                outTrain.close();
                outTest.flush();
                outTest.close();

            }catch(FileNotFoundException | NumberFormatException e){
                System.out.println("Fold " + foldNumber + " not present: "+ e);
            }
        }else{
            System.out.println(dir.getAbsolutePath() + ": Already exists.");
        }
        
        
    } 

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        classifyInstance(instance);
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}

