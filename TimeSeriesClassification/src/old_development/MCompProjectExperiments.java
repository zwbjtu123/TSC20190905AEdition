package old_development;

import statistics.simulators.DataSimulator;
import statistics.simulators.ShapeletModel;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;
import weka.filters.timeseries.shapelet_transforms.ApproximateShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.CachedSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.OnlineSubSeqDistance;

/**
 *
 * @author Edgaras
 */
public class MCompProjectExperiments {
    public static String dropboxPath="C:\\Users\\Edgaras\\Dropbox\\TSC Problems";
        
    // There are two types of dataset assessment - LOOCV or Train/Test split
    private enum AssesmentType{LOOCV, TRAIN_TEST};
      
    public static String[] fileNames={
                                    //Number of train,test cases,length,classes,total num of datapoints
        "SonyAIBORobotSurface",     //20,601,70,2,1400
        "ItalyPowerDemand",         //67,1029,24,2,1608
        "MoteStrain",               //20,1252,84,2,1680
        "TwoLeadECG",               //23,1139,82,2,1886
        "ECGFiveDays",              //23,861,136,2,3128
        "DiatomSizeReduction",      //16,306,345,4,5520
        "GunPoint",                 //50,150,150,2,7500
        "Coffee",                   //28,28,286,2,8008
        "FaceFour",                 //24,88,350,4,8400
        "Symbols",                  //25,995,398,6,9950
        "Beef",                     //30,30,470,5,14100
        "SyntheticControl",         //300,300,60,6,18000
        "MPEG7Shapes/beetle-fly",   //40,,512,2,20480
        "MPEG7Shapes/bird-chicken", //40,,512,2,20480
        "Lighting7",                //70,73,319,7,22330
        "Trace",                    //100,100,275,4,27500
        "otoliths/Herrings",        //64,64,512,2,32768
        "MedicalImages",            //381,760,99,10,37719
        "otoliths/Herring500",      //100,,500,2,50000
        "SyntheticData",            //100,1000,500,2,50000
        "Adiac",                    //390,391,176,37,68640
        "ChlorineConcentration",    //467,3840,166,3,77522
        "Bones/DP_Little",          //400,645,250,3,100000
        "Bones/DP_Middle",          //400,645,250,3,100000
        "Bones/DP_Thumb",           //400,645,250,3,100000
        "Bones/MP_Little",          //400,645,250,3,100000
        "Bones/MP_Middle",          //400,645,250,3,100000
        "Bones/PP_Little",          //400,645,250,3,100000
        "Bones/PP_Middle",          //400,645,250,3,100000
        "Bones/PP_Thumb",           //400,645,250,3,100000
        "MPEG7Shapes/ShapesAll",    //600,600,512,60,307200
    };
    
    // An array containing the assesment type for each of the datasets. 
    private static MCompProjectExperiments.AssesmentType[] assesmentTypes = {
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //SonyAIBORobotSurface
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //ItalyPowerDemand
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //MoteStrain
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //TwoLeadECG
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //ECGFiveDays
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //DiatomSizeReduction
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //GunPoint
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Coffee
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //FaceFour
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Symbols
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Beef
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //SyntheticControl
        MCompProjectExperiments.AssesmentType.LOOCV,      //MPEG7Shapes/beetle-fly
        MCompProjectExperiments.AssesmentType.LOOCV,      //MPEG7Shapes/bird-chicken
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Lighting7
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Trace
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //otoliths/Herrings
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //MedicalImages
        MCompProjectExperiments.AssesmentType.LOOCV,      //otoliths/Herring500
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //SyntheticData
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Adiac
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //ChlorineConcentration
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/DP_Little
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/DP_Middle
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/DP_Thumb
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/MP_Little
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/MP_Middle
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/PP_Little
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/PP_Middle
        MCompProjectExperiments.AssesmentType.TRAIN_TEST, //Bones/PP_Thumb
        MCompProjectExperiments.AssesmentType.TRAIN_TEST  //MPEG7Shapes/ShapesAll
    };
    
    // An array containing the shapelet min-max interval for each of the datasets. 
    private static int[][] shapeletMinMax = {
        {15, 36},   // SonyAIBORobotSurface
        {7, 14},    // ItalyPowerDemand
        {16, 31},   // MoteStrain
        {7, 13},    // TwoLeadECG
        {24, 76},   // ECGFiveDays
        {7,16},     // DiatomSizeReduction
        {24, 55},   // GunPoint
        {18,30},    // Coffee
        {20, 120},  // FaceFour
        {52, 155},  // Symbols
        {8, 30},    // Beef
        {20, 56},   // SyntheticControl
        {30, 101},  // MPEG7Shapes/beetle-fly
        {30, 101},  // MPEG7Shapes/bird-chicken
        {20, 80},   // Lighting7
        {62, 232},  // Trace
        {30, 101},  // otoliths/Herrings
        {9, 35},    // MedicalImages
        {30, 101},  // otliths/Herring500
        {25, 35},   // SyntheticData
        {3, 10},    // Adiac
        {7, 20},    // ChlorineConcentration
        {9, 36},    // Bones/DP_Little 
        {15, 43},   // Bones/DP_Middle
        {11, 47},   // Bones/DP_Thumb        
        {15, 41},   // Bones/MP_Little
        {20, 53},   // Bones/MP_Middle
        {13, 38},   // Bones/PP_Little        
        {14, 34},   // Bones/PP_Middle
        {14, 41},   // Bones/PP_Thumb        
        {30, 110}   // MPEG7Shapes/ShapesAll        
    };
    
    
    // Variables for holding data
    private static Instances[] instancesTrain;
    private static Instances[] instancesTest;
    
    // SYNTHETIC DATA PARMAETERS
    private static final int STARTING_LENGTH = 50;
    private static final int MAX_LENGTH = 500;
    private static final int STEP_SIZE = 50;
    private static final int NUM_OF_CASES = 100;
    private static final int SHAPELET_LENGTH = 30;
       
    // APPROXIMATION PARAMETERS
    // Note: PERCENT_END - PERCENT_START % PERCENT_INCREMENT == 0
    private static final int PERCENT_START = 45;
    private static final int PERCENT_END = 95;
    private static final int PERCENT_INCREMENT = 10;
    private static final int UCR_NUM_FILES = 31;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        loadData();
        
        //Experiment selection
        Scanner scan = new Scanner(System.in);
        int experimentIndex = 0;
        boolean isValidInput = false;
        
        System.out.println("Select experiment: "
                //Preparatory experiments
                + "\n \t 1. Estimate min/max (Note: Only used once and then values are hardcoded).\n"
                
                //Accuracy experiments
                + "\n \t 2. Compare C4.5 accuracy of different subsequence distance methods using UCR data."
                + "\n \t 3. Compare C4.5 accuracy of transform (DistCaching) with and without candidate IG pruning using UCR data."
                + "\n \t 4. Compare C4.5 accuracy of transform (DistCaching) with and without candidate MM pruning using UCR data."
                + "\n \t 5. Compare C4.5 accuracy of transform (DistCaching) with and without candidate F-Stat pruning using UCR data."
                + "\n \t 6. Compare C4.5 accuracy of transform (DistCaching) with and without candidate KW pruning using UCR data."
                + "\n \t 7. Compare C4.5 accuracy of transform (DistCaching) with and without candidate IG pruning using synthetic data."
                + "\n \t 8. Compare C4.5 accuracy of transform (DistCaching) with and without candidate MM pruning using synthetic data."
                + "\n \t 9. Compare C4.5 accuracy of transform (DistCaching) with and without candidate F-Stat pruning using synthetic data."
                + "\n \t 10. Compare C4.5 accuracy of transform (DistCaching) with and without candidate KW pruning using synthetic data.\n"
                
                //Timing experiments
                + "\n \t 11. Compare speed of different subsequence distance methods using UCR data."
                + "\n \t 12. Compare speed of different subsequence distance methods using synthetic data."
                + "\n \t 13. Compare speed of transform (DistCaching) with and without candidate IG pruning using UCR data."
                + "\n \t 14. Compare speed of transform (DistCaching) with and without candidate MM pruning using UCR data."
                + "\n \t 15. Compare speed of transform (DistCaching) with and without candidate F-Stat pruning using UCR data."
                + "\n \t 16. Compare speed of transform (DistCaching) with and without candidate KW pruning using UCR data."
                + "\n \t 17. Compare speed of transform (DistCaching) with and without candidate IG pruning using synthetic data."
                + "\n \t 18. Compare speed of transform (DistCaching) with and without candidate MM pruning using synthetic data."
                + "\n \t 19. Compare speed of transform (DistCaching) with and without candidate F-Stat pruning using synthetic data."
                + "\n \t 20. Compare speed of transform (DistCaching) with and without candidate KW pruning using synthetic data.\n"
                
                //Operation count experiments
                + "\n \t 21. Compare fundamental operation count of base transform vs transform with online normalization and reordering.\n"
                
                //Approximate transform experiments
                + "\n \t 22. Compare C4.5 accuracy of exact and approximate transforms using UCR data."
                + "\n \t 23. Compare C4.5 accuracy of exact and approximate transforms using synthetic data."
                + "\n \t 24. Compare speed of exact and approximate transforms using UCR data."
                + "\n \t 25. Compare speed of exact and approximate transforms using synthetic data");
        while(!isValidInput){
            try{
                int in = scan.nextInt();
                if(in > 0 && in < 26){
                    experimentIndex = in;
                    isValidInput = true;
                }else{
                    throw new IOException();
                }
            }catch(Exception e){
                scan = new Scanner(System.in);
                System.out.println("Invalid experiment selection.");
            }
        }
        
        switch (experimentIndex){
            //Preparatory experiments
            case 1: estimateMinMaxExperiment(); break;
                
            //Accuracy experiments
            case 2: exactTransformAccuracyExperiment(); break;
            case 3: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN, false); break;    
            case 4: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.MOODS_MEDIAN, false); break; 
            case 5: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.F_STAT, false); break; 
            case 6: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.KRUSKALL_WALLIS, false); break;
            case 7: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN, true); break;    
            case 8: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.MOODS_MEDIAN, true); break; 
            case 9: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.F_STAT, true); break; 
            case 10: candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice.KRUSKALL_WALLIS, true); break;       
            
            //Timing experiments    
            case 11: exactTransformTimingExperiment(false); break;
            case 12: exactTransformTimingExperiment(true); break;    
            case 13: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN, false); break;
            case 14: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.MOODS_MEDIAN, false); break;
            case 15: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.F_STAT, false); break;
            case 16: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.KRUSKALL_WALLIS, false); break;
            case 17: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN, true); break;
            case 18: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.MOODS_MEDIAN, true); break;
            case 19: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.F_STAT, true); break;
            case 20: candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice.KRUSKALL_WALLIS, true); break;
                
            //Operation count experiments
            case 21: opCountForBaseAndOnlineTransform(); break;
            
            //Approximate transform experiments
            case 22: approxTransformAccuracyExperiment(false); break;
            case 23: approxTransformAccuracyExperiment(true); break;
            case 24: approxTransformTimingExperiment(false); break;   
            case 25: approxTransformTimingExperiment(true); break; 
            default: System.out.println("Unknow experiment identifier.");
        }
    }
    
    //################### Experiment Functions #################################
    
    // Method to estimate min/max (Note: Only used once and then values are hardcoded).
    private static void estimateMinMaxExperiment(){
        //Prepare output file
        String fileName = "estimated_min_max.csv";
        String content = "Dataset, Min, Max";
        writeToFile(fileName, content, false);
        
        //Find min/max
        for(int i = 0; i < instancesTrain.length; i++){
            System.out.println("Processing dataset " + (i+1) +" out of " + instancesTrain.length);
            int[] minMax = estimateMinAndMax(instancesTrain[i]);
            content = fileNames[i] + ", " + minMax[0] + ", " + minMax[1];
            writeToFile(fileName, content, true);
        }
    }

    //Method to check C4.5 accuracy of different subsequence distance methods.
    private static void exactTransformAccuracyExperiment(){
        
        Classifier classifier = new J48();
        
        //Prepare output file
        String fileName = "subsequence_distance_UCR_accuracy.csv";
        String content = "Dataset, C4.5 using Base transform, C4.5 using online z-norm + reordering, C4.5 using stat caching";
        writeToFile(fileName, content, false);
        
        // Record filter times to find single shapelet
        for(int i = 0; i < instancesTrain.length; i++){
            System.out.println("Processing dataset " + (i+1) +" out of " + instancesTrain.length);
            FullShapeletTransform[] transforms = initExactTransformExperimentTransforms();
            ucrDataAccuracy(classifier, transforms, i, fileName);
        }
    }
    
    //Method to check C4.5 accuracy of transforms with and without candidate pruning
    private static void candidatePruningAccuracyExperiment(QualityMeasures.ShapeletQualityChoice qualityChoice, boolean useSyntheticData){
        
        Classifier classifier = new J48();
               
        //Prepare output file
        String fileName;
        if(useSyntheticData){
            fileName = "candidate_pruning_accuracy_synthetic_"+qualityChoice+".csv";
        }else{
            fileName = "candidate_pruning_accuracy_UCR_"+qualityChoice+".csv";
        }
        String content = "Dataset, C4.5 using transform without candidate pruning, C4.5 using transform with candidate pruning";
        writeToFile(fileName, content, false);
        
        FullShapeletTransform[] transforms;
        if(useSyntheticData){
            for(int seriesLength = STARTING_LENGTH; seriesLength <= MAX_LENGTH; seriesLength += STEP_SIZE){
                System.out.println("Processing length " + seriesLength+ " out of " + MAX_LENGTH);
                transforms = initCandidatePruningExperimentTransforms(qualityChoice);
                Instances[] syntheticData = generateData(NUM_OF_CASES, SHAPELET_LENGTH, seriesLength);
                syntheticDataAccuracy(classifier, transforms, syntheticData, seriesLength, fileName);
            }
        }else{
            for(int i = 0; i < instancesTrain.length; i++){
                System.out.println("Processing dataset " + (i+1) +" out of " + instancesTrain.length);
                transforms = initCandidatePruningExperimentTransforms(qualityChoice);
                ucrDataAccuracy(classifier, transforms, i, fileName);
            }
        }
    }
    
    //Method to perform timing experiment on base and optimized shapelet transforms. 
    //The time taken to find a single best shapelet from the dataset is recorded.
    private static void exactTransformTimingExperiment(boolean useSyntheticData){
        
        // Initialise transforms required for this experiment
        FullShapeletTransform[] transforms = initExactTransformExperimentTransforms();
        
        //Prepare output file
        String fileName;
        if(useSyntheticData){
            fileName = "subsequence_distance_synthetic_timing.csv";
        }else{
            fileName = "subsequence_distance_UCR_timing.csv";
        }
        
        //Prepare output file if one does not exist
        //if(!isFileExists(fileName)){
        String content = "Dataset, Base, Online z-norm + reordering, Stat caching";
        writeToFile(fileName, content, false);
        //}
        
        if(useSyntheticData){
            // Record transform times to find single shapelet           
            for(int seriesLength = STARTING_LENGTH; seriesLength <= MAX_LENGTH; seriesLength +=STEP_SIZE){
                System.out.println("Processing length " + seriesLength+ " out of " + MAX_LENGTH);
                Instances[] syntheticData = generateData(NUM_OF_CASES, SHAPELET_LENGTH, seriesLength);
                runTiming(transforms, syntheticData[0], String.valueOf(seriesLength), SHAPELET_LENGTH, SHAPELET_LENGTH, fileName);
            }
        }else{      
            // Record filter times to find single shapelet
            for(int i = 0; i < instancesTrain.length; i++){
                System.out.println("Processing dataset " + (i+1) +" out of " + instancesTrain.length);
                runTiming(transforms, instancesTrain[i], fileNames[i], shapeletMinMax[i][0], shapeletMinMax[i][1], fileName);
            }
        }
    }
    
    //Method to perform timing experiment on transform with and withou candidate pruning. 
    //The time taken to find a single best shapelet from the dataset is recorded.
    private static void candidatePruningTimingExperiment(QualityMeasures.ShapeletQualityChoice qualityChoice, boolean useSyntheticData){
        
        // Initialise transforms required for this experiment
        FullShapeletTransform[] transforms = initCandidatePruningExperimentTransforms(qualityChoice);
        
        //Prepare output file
        String fileName;
        if(useSyntheticData){
            fileName = "candidate_pruning_timing_synthetic_"+qualityChoice+".csv";
        }else{
            fileName = "candidate_pruning_timing_UCR_"+qualityChoice+".csv";
        }
        String content = "Dataset, Dist Without pruning, With pruning";
        writeToFile(fileName, content, false);
          
        if(useSyntheticData){
            // Record transform times to find single shapelet           
            for(int seriesLength = STARTING_LENGTH; seriesLength <= MAX_LENGTH; seriesLength +=STEP_SIZE){
                System.out.println("Processing length " + seriesLength+ " out of " + MAX_LENGTH);
                Instances[] syntheticData = generateData(NUM_OF_CASES, SHAPELET_LENGTH, seriesLength);
                runTiming(transforms, syntheticData[0], String.valueOf(seriesLength), SHAPELET_LENGTH, SHAPELET_LENGTH, fileName);
            }
        }else{
            for(int i = 0; i < instancesTrain.length; i++){
                System.out.println("Processing dataset " + (i+1) +" out of " + instancesTrain.length);
                runTiming(transforms, instancesTrain[i], fileNames[i], shapeletMinMax[i][0], shapeletMinMax[i][1], fileName);
            }
        }
    }
    
    //Method to perform fundamental operation count experiment on base and 
    //optimised transform which performs online normalisation and reordering.
    private static void opCountForBaseAndOnlineTransform(){
        
        // Initialise filters required for this experiment
        FullShapeletTransform[] transforms = new FullShapeletTransform[2];
        transforms[0] = new FullShapeletTransform();
        transforms[1] = new FullShapeletTransform();
        transforms[1].setSubSeqDistance(new OnlineSubSeqDistance());

        transforms[0].turnOffLog();
        transforms[0].supressOutput();

        transforms[1].turnOffLog();
        transforms[1].supressOutput();
        
        //Prepare output file
        String fileName = "subseq_dist_op_count_base_and_online_transform.csv";
        String content = "Dataset, Base, Online + reordering";
        writeToFile(fileName, content, false);
        
        // Record filter times to find single shapelet
        StringBuilder sb;
        for(int seriesLength = STARTING_LENGTH; seriesLength <= MAX_LENGTH; seriesLength += STEP_SIZE){
            System.out.println("Processing length " + seriesLength);
            sb = new StringBuilder();
            sb.append(seriesLength);
            sb.append(", ");
            for(int j = 0; j < transforms.length; j++){
                try{
                    Instances[] syntheticData = generateData(NUM_OF_CASES, SHAPELET_LENGTH, seriesLength);
                    long count = transforms[j].opCountForSingleShapelet(syntheticData[0], SHAPELET_LENGTH, SHAPELET_LENGTH);
                    if(count < 0){
                        System.out.println("Overflow!");
                    }
                    sb.append(count);
                }catch(Exception ex){
                    Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
                }
                if(j != transforms.length-1){
                    sb.append(", ");
                }
            }
            writeToFile(fileName, sb.toString(), true);
        }
    }
    
    
    //Method to check C4.5 accuracy of base and approximate shapelet transforms.
    private static void approxTransformAccuracyExperiment(boolean useSyntheticData){
        
        Classifier classifier = new J48();
               
        //Prepare output file
        String fileName;
        if(useSyntheticData){
            fileName = "approx_transform_accuracies_synthetic.csv";
        }else{
            fileName = "approx_transform_accuracies_UCR.csv";
        }
        initApproxTransformExperimentFile(fileName);

        //Run experiment
        FullShapeletTransform[] transforms;
        if(useSyntheticData){
            for(int seriesLength = STARTING_LENGTH; seriesLength <= MAX_LENGTH; seriesLength += STEP_SIZE){
                System.out.println("Processing length " + seriesLength+ " out of " + MAX_LENGTH);

                //Initialise filters required for this experiment
                transforms = initApproxTransformExperimentTransforms();

                Instances[] syntheticData = generateData(NUM_OF_CASES, SHAPELET_LENGTH, seriesLength);

                syntheticDataAccuracy(classifier, transforms, syntheticData, seriesLength, fileName);
            }
        }else{
            for(int i = 15; i < UCR_NUM_FILES; i++){
                System.out.println("Processing dataset " + (i+1) +" out of " + instancesTrain.length);

                //Initialise filters required for this experiment
                transforms = initApproxTransformExperimentTransforms();

                //Write accuraries
                ucrDataAccuracy(classifier, transforms, i, fileName);
            }
        }
        
    }
    
    //Method to perform timing experiment on base and approximate shapelet transforms. 
    //The time taken to find a single best shapelet from the dataset is recorded.
    private static void approxTransformTimingExperiment(boolean useSyntheticData){
       
            // Initialise filters required for this experiment
            FullShapeletTransform[] transforms = initApproxTransformExperimentTransforms();

            //Prepare output file
            String fileName;
            if(useSyntheticData){
                fileName = "approx_transform_timing_synthetic.csv";
            }else{
                fileName = "approx_transform_timing_UCR.csv";
            }
            initApproxTransformExperimentFile(fileName);

            if(useSyntheticData){
                for(int seriesLength = STARTING_LENGTH; seriesLength <= MAX_LENGTH; seriesLength +=STEP_SIZE){
                    System.out.println("Processing length " + seriesLength+ " out of " + MAX_LENGTH);
                    Instances[] syntheticData = generateData(NUM_OF_CASES, SHAPELET_LENGTH, seriesLength);
                    runTiming(transforms, syntheticData[0], String.valueOf(seriesLength), SHAPELET_LENGTH, SHAPELET_LENGTH, fileName);
                }
            }else{
                for(int i = 22; i < UCR_NUM_FILES; i++){
                    System.out.println("Processing dataset " + (i+1) +" out of " + instancesTrain.length);
                    runTiming(transforms, instancesTrain[i], fileNames[i], shapeletMinMax[i][0], shapeletMinMax[i][1], fileName);
                }
            }
        
    }
    
    //################### End of Experiment Functions ##########################

    
    
    
    
    //################### Helper Functions #####################################
    
    //Class implementing comparator which compares shapelets according to their length
    public static class ShapeletLengthComparator implements Comparator{
   
        @Override
        public int compare(Object shapelet1, Object shapelet2){

            int shapelet1Length = ((Shapelet)shapelet1).getContent().length;        
            int shapelet2Lenght = ((Shapelet)shapelet2).getContent().length;

            if(shapelet1Length > shapelet2Lenght) {
                return 1;
            }else if(shapelet1Length < shapelet2Lenght) {
                return -1;
            }else {
                return 0;
            }    
        }
    }
    
    //Method to prepare output file for approximate transform experiments
    private static void initApproxTransformExperimentFile(String fileName){
        //Prepare output file
        StringBuilder sb = new StringBuilder();
        sb.append("Dataset, ");
        sb.append("Base, ");
        for(int percentage = PERCENT_END; percentage >= PERCENT_START; percentage -= PERCENT_INCREMENT){
            sb.append("Approx_");
            sb.append(percentage);
            if(percentage > PERCENT_START){
                sb.append(", ");
            }
        }
        
        if(!isFileExists(fileName)){
            String content = "Dataset, Base, Online z-norm + reordering, Stat caching";
            writeToFile(fileName, sb.toString(), false);
        }
        
    }
    
    //Method to initialise transforms for exact transform experiments
    private static FullShapeletTransform[] initExactTransformExperimentTransforms(){
        FullShapeletTransform[] transforms = new FullShapeletTransform[3];
        transforms[0] = new FullShapeletTransform();
        transforms[1] = new FullShapeletTransform();
        transforms[1].setSubSeqDistance(new OnlineSubSeqDistance());
        transforms[2] = new FullShapeletTransform();
        transforms[2].setSubSeqDistance(new CachedSubSeqDistance());
        
        transforms[0].turnOffLog();
        transforms[0].supressOutput();
        
        transforms[1].turnOffLog();
        transforms[1].supressOutput();
        
        transforms[2].turnOffLog();
        transforms[2].supressOutput();
        
        return transforms;
    }
    
    //Method to initialise transforms for candidate pruning experiments
    private static FullShapeletTransform[] initCandidatePruningExperimentTransforms(QualityMeasures.ShapeletQualityChoice qualityChoice){
        // Initialise transforms required for this experiment
        FullShapeletTransform[] transforms = new FullShapeletTransform[2];
        transforms[0] = new FullShapeletTransform();
        transforms[0].setSubSeqDistance(new CachedSubSeqDistance());
        transforms[1] = new FullShapeletTransform();
        transforms[1].setSubSeqDistance(new CachedSubSeqDistance());
        
        transforms[0].turnOffLog();
        transforms[0].supressOutput();
        
        transforms[1].turnOffLog();
        transforms[1].supressOutput();
        transforms[1].setQualityMeasure(qualityChoice);
        transforms[1].useCandidatePruning();
        
        return transforms;
    }
            
    //Method to initialise transforms for approximate transform experiments
    private static FullShapeletTransform[] initApproxTransformExperimentTransforms(){
   
        if((PERCENT_END - PERCENT_START) % PERCENT_INCREMENT != 0){
            System.err.println("Incorrect approximation parameters");
            System.exit(0);
        }
        
        int numOfTransforms = ((PERCENT_END - PERCENT_START) / PERCENT_INCREMENT) + 2;
        
        FullShapeletTransform[] transforms = new FullShapeletTransform[numOfTransforms];
        transforms[0] = new FullShapeletTransform();
        transforms[0].setSubSeqDistance(new CachedSubSeqDistance());
        transforms[0].turnOffLog();
        transforms[0].supressOutput();
            
        int transformIndex = 1;
        for(int percentage = PERCENT_END; percentage >= PERCENT_START; percentage -= PERCENT_INCREMENT){
            try {
                //Initialise approx transforms required for the experiment
                ApproximateShapeletTransform ast = new ApproximateShapeletTransform();
                ast.setSampleLevels(percentage, percentage);
                transforms[transformIndex] = ast;
                transforms[transformIndex].turnOffLog();
                transforms[transformIndex].supressOutput();
                transformIndex++;
            } catch (IOException ex) {
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return transforms;
    }
    
    // Method to load the datasets.
    private static void loadData(){
        instancesTrain = new Instances[fileNames.length];
        instancesTest = new Instances[fileNames.length];
            
        //Load all the datasets and set class index for loaded instances
        for(int i=0; i<fileNames.length; i++){
            
            String dir;
            String fileName;
            String[] splits = null;
            if(fileNames[i].contains("/")){
                splits = fileNames[i].split("/");
            }
            
            if(splits != null){
                dir = splits[0];
                fileName = splits[1];
            }else{
                dir = fileNames[i];
                fileName = fileNames[i];
            }
            
            // Load test/train splits
            if(assesmentTypes[i] == MCompProjectExperiments.AssesmentType.TRAIN_TEST){
                instancesTrain[i] = FullShapeletTransform.loadData(dropboxPath+"\\"+dir+"\\"+fileName+"_TRAIN.arff");
                instancesTest[i] = FullShapeletTransform.loadData(dropboxPath+"\\"+dir+"\\"+fileName+"_TEST.arff");
            }else if(assesmentTypes[i] == MCompProjectExperiments.AssesmentType.LOOCV){
                instancesTrain[i] = FullShapeletTransform.loadData(dropboxPath+"\\"+dir+"\\"+fileName+".arff");
                instancesTest[i] = null;   
            }
            
            // Set class indices
            instancesTrain[i].setClassIndex(instancesTrain[i].numAttributes() - 1);
            if(assesmentTypes[i] == MCompProjectExperiments.AssesmentType.TRAIN_TEST){
                instancesTest[i].setClassIndex(instancesTest[i].numAttributes() - 1);
            }
        }     
    }
    
    // Method to estimate min/max shapelet lenght for a given data
    private static int[] estimateMinAndMax(Instances data){
        ArrayList<Shapelet> shapelets = new ArrayList<Shapelet>();
        FullShapeletTransform st = new FullShapeletTransform();
        st.setSubSeqDistance(new CachedSubSeqDistance());
        st.supressOutput();
        st.turnOffLog();
          
        Instances randData =  new Instances(data);
        Instances randSubset;
        
        for(int i = 0; i < 10; i++){
            randData.randomize(new Random());
            randSubset = new Instances(randData, 0, 10);
            try{
                shapelets.addAll(st.findBestKShapeletsCache(10, randSubset, 1, randSubset.numAttributes()-1));
            }catch(Exception e){
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, e);
            } 
        }
        
        /*
        //So instead I just select 100 best shapelet from the training data
        //which is an overkill but will yield good shapelet lengths
        try{
            shapelets.addAll(st.findBestKShapeletsCache(100, data, 1, data.numAttributes()-1));
        }catch(Exception ex){
            Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
        } */
                
        Collections.sort(shapelets, new ShapeletLengthComparator());
        int min = shapelets.get(24).getContent().length;
        int max = shapelets.get(74).getContent().length;
        
        int[] parEstimates = {min, max};
        
        return parEstimates;
    }
    
    //Method to perform accuracy test for all transforms on a given synthetic dataset
    private static void syntheticDataAccuracy(Classifier classifier, FullShapeletTransform[] transforms, Instances[] syntheticData, int seriesLength, String fileName){
                
        StringBuilder sb = new StringBuilder();
        sb.append(seriesLength);
        sb.append(", ");
        for(int j = 0; j < transforms.length; j++){
            transforms[j].setNumberOfShapelets((syntheticData[0].numAttributes()-1)/2);
            transforms[j].setShapeletMinAndMax(SHAPELET_LENGTH, SHAPELET_LENGTH);

            try{
                Instances tempTrain = instancesTrain[0];
                Instances tempTest = instancesTest[0];
                instancesTrain[0] = syntheticData[0];
                instancesTest[0] = syntheticData[1];
                sb.append(classifierAccuracy(classifier, 0, transforms[j], false, true));
                instancesTrain[0] = tempTrain;
                instancesTest[0] = tempTest;
            }catch(Exception ex){
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
            if(j != transforms.length-1){
                sb.append(", ");
            }
        }
        writeToFile(fileName, sb.toString(), true);
    }
    
    //Method to perform accuracy test for all transforms on a given UCR dataset
    private static void ucrDataAccuracy(Classifier classifier, FullShapeletTransform[] transforms, int dataIndex, String fileName){
        //Get accuracies
        StringBuilder sb = new StringBuilder();
        sb.append(fileNames[dataIndex]);
        sb.append(", ");
        for(int j = 0; j < transforms.length; j++){
            transforms[j].setNumberOfShapelets((instancesTrain[dataIndex].numAttributes()-1)/2);
            transforms[j].setShapeletMinAndMax(shapeletMinMax[dataIndex][0], shapeletMinMax[dataIndex][1]);
            try{
                sb.append(classifierAccuracy(classifier, dataIndex, transforms[j], false, true));
            }catch(Exception ex){
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
            if(j != transforms.length-1){
                sb.append(", ");
            }
        }
        writeToFile(fileName, sb.toString(), true);
    }
    
    private static void runTiming(FullShapeletTransform[] transforms, Instances data, String dataSetName,int minShapeletLength, int maxShapeletLength, String fileName){
        StringBuilder sb = new StringBuilder();
        sb.append(dataSetName);
        sb.append(", ");
        for(int j = 0; j < transforms.length; j++){
            try{
                double time = transforms[j].timingForSingleShapelet(data, minShapeletLength, maxShapeletLength);
                sb.append(time);
            }catch(Exception ex){
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
            if(j != transforms.length-1){
                sb.append(", ");
            }
        }
        writeToFile(fileName, sb.toString(), true);
    }
    
    // Method to validate a given classifier
    private static double classifierAccuracy(Classifier classifier, 
                                             int dataIndex, 
                                             FullShapeletTransform transform, 
                                             boolean computeErrorRate, 
                                             boolean usePercentage){

        double accuracy = 0.0;
        
        //Generate accuray
        if(assesmentTypes[dataIndex] == AssesmentType.TRAIN_TEST){
            accuracy = classifierAccuracyTrainTest(classifier, dataIndex, transform);
        }else if(assesmentTypes[dataIndex] == AssesmentType.LOOCV){
            accuracy = classifierAccuracyLOOCV(classifier, dataIndex, transform);
        }

        if(computeErrorRate){
            accuracy = 1 - accuracy;
        }

        if(usePercentage){
            accuracy *= 100;
        }       
    
             
        return accuracy;
    }
    
    //Method to perform simple train/test split validation using given classifier
    private static double classifierAccuracyTrainTest(Classifier classifier, int dataIndex, FullShapeletTransform transform){

        double accuracy = 0.0;
        
        Instances trainData = null, testData = null;
        
        if(transform != null){            
            //Transform data
            try{
                trainData = transform.process(instancesTrain[dataIndex]);
                testData = transform.process(instancesTest[dataIndex]);
            }catch(Exception ex){
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
        }else{
            trainData = instancesTrain[dataIndex];
            testData = instancesTest[dataIndex];
        }   
        
        try {
            classifier.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
        }

        //Classify test instancs while recording accuracy
        for(int j = 0; j < testData.numInstances(); j++){

            double classifierPrediction = 0.0;
            try{
                classifierPrediction = classifier.classifyInstance(testData.instance(j));
            }catch(Exception ex){
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            double actualClass = testData.instance(j).classValue();

            if(classifierPrediction == actualClass) {
                accuracy++;
            }

            // Compute average accuracy if it is the last test instance
            if(j == testData.numInstances() - 1){
                accuracy /= testData.numInstances();
            }
        }
        
        return accuracy;
    }
    
    //Method to perform leave one out cross validation using given classifier and
    private static double classifierAccuracyLOOCV(Classifier classifier, int dataIndex, FullShapeletTransform transform){
               
        //Variables for holding folds
        Instances data = instancesTrain[dataIndex];
        Instances trainFold;
        Instances testFold;
        
        double accuracy = 0.0;
        
        //Generate average accuracies
        for (int n = 0; n < data.numInstances(); n++) {
            System.out.println("Processing fold: " + n);
            //Generate folds
            trainFold = data.trainCV(data.numInstances(), n);
            testFold = data.testCV(data.numInstances(), n);
  
            if(transform != null){            
                //Transform data
                try{
                    trainFold = transform.process(trainFold);
                    testFold = transform.process(testFold);
                }catch(Exception ex){
                    Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            
            try {
                classifier.buildClassifier(trainFold);
            } catch (Exception ex) {
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }

            double classifierPrediction = 0.0;
            try{
                classifierPrediction = classifier.classifyInstance(testFold.instance(0));
            }catch(Exception ex){
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            double actualClass = testFold.instance(0).classValue();

            if(classifierPrediction == actualClass) {
                accuracy++;
            }
            
            // Compute average accuracy if it is the last test instance
            if(n == data.numInstances() - 1){
                accuracy /= data.numInstances();
            }
        }

        return accuracy;
    }
    
    //Method to write text into a file.
    private static void writeToFile(String filename, String text, boolean append) {
        
        BufferedWriter bufferedWriter = null;
        
        try {       
            //Construct the BufferedWriter object
            bufferedWriter = new BufferedWriter(new FileWriter(filename, append));
            
            //Start writing to the output stream
            bufferedWriter.write(text);
            bufferedWriter.newLine();
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            //Close the BufferedWriter
            try {
                if (bufferedWriter != null) {
                    bufferedWriter.flush();
                    bufferedWriter.close();
                }
            } catch (IOException ex) {
                Logger.getLogger(MCompProjectExperiments.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
 
    //Method to check if file with a given name exists.
    private static boolean isFileExists(String filename){
        File f = new File(filename);
        if(f.isFile() && f.canWrite()) {
            return true;
        }else{
            return false;
        }   
    }
  
    //Method to generate synthetic data
    private static Instances[] generateData(int nosCases, int shapeletLength, int seriesLength){
        ShapeletModel[] s=new ShapeletModel[2];
        int[] casesPerClass={nosCases/2,nosCases/2};
        
        //PARAMETER LIST:  numShapelets, seriesLength, shapeletLength, maxStart        
        double[] p1={1,seriesLength,shapeletLength};
        double[] p2={1,seriesLength,shapeletLength};
        
        //Create two ShapeleModels with different base Shapelets        
        s[0]=new ShapeletModel(p1);        
        ShapeletModel.ShapeType st=s[0].getShapeType();
        s[1]=new ShapeletModel(p2);
        while(st==s[1].getShapeType()){
            s[1]=new ShapeletModel(p2);
        }
            
        //System.out.println(" Shape 1= "+s[0]);
        //System.out.println(" Shape 2= "+s[1]);
        
        DataSimulator ds=new DataSimulator(s);
        Instances train=ds.generateDataSet(seriesLength,casesPerClass);
        Instances test=ds.generateDataSet(seriesLength,casesPerClass);
        Instances[] output = {train, test};
        return output;
    }
    //################### End of Helper Functins ###############################
}
