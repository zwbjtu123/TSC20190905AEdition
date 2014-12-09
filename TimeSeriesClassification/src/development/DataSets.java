/*
 Class containing data sets used in various TSC papers
 */
package development;

/**
 *
 * @author ajb
 */
public class DataSets {
  //ALL of our data sets  
    //<editor-fold defaultstate="collapsed" desc="fileNames: All Data sets">    
		public static String[] fileNames={	
			"Adiac", // 390,391,176,37
			"ArrowHead", // 36,175,251,3
			"Beef", // 30,30,470,5
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinC_ECG_torso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes", // 322,139,512,2
			"ECGFiveDays", // 23,861,136,2
			"ElectricDevices", // 8926,7711,96,7
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fiftywords", // 450,455,270,50
			"fish", // 175,175,463,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"GunPoint", // 50,150,150,2
			"Ham",
                        "HandOutlines", // 1000,370,2709,2
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"MALLAT", // 55,2345,1024,8
			"Meat",
                        "MedicalImages", // 381,760,99,10
			"MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
			"NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"PhalangesOutlinesCorrect", // 1800,858,80,2
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"ShapesAll", // 600,600,512,60
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface", // 20,601,70,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"StarLightCurves", // 1000,8236,1024,3
			"Strawberry",
                        "SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"wafer", // 1000,6164,152,2
			"Wine",
                        "WordSynonyms", // 267,638,270,25
			"Worms",
                        "WormsTwoClass",
                        "yoga" // 300,3000,426,2
                };   
      //</editor-fold>    
    
//UCR data sets
    //<editor-fold defaultstate="collapsed" desc="fileNames: 46 UCR Data sets">    
		public static String[] ucrNames={	
			"Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinC_ECG_torso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"ECGFiveDays", // 23,861,136,2
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fiftywords", // 450,455,270,50
			"fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"Haptics", // 155,308,1092,5
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"MALLAT", // 55,2345,1024,8
                        "MedicalImages", // 381,760,99,10
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
			"NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"Plane", // 105,105,144,7
			"SonyAIBORobotSurface", // 20,601,70,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"StarLightCurves", // 1000,8236,1024,3
                        "SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"wafer", // 1000,6164,152,2
                        "WordSynonyms", // 267,638,270,25
                        "yoga" // 300,3000,426,2
                };   
      //</editor-fold>

//Small UCR data sets
    //<editor-fold defaultstate="collapsed" desc="fileNames: Small UCR Data sets">    
		public static String[] ucrSmall={	
			"Beef", // 30,30,470,5
			"Car", // 60,60,577,4
			"Coffee", // 28,28,286,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"OliveOil", // 30,30,570,4
			"Plane", // 105,105,144,7
			"SonyAIBORobotSurface", // 20,601,70,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
                };   
      //</editor-fold>



//<editor-fold defaultstate="collapsed" desc="rakthanmanon13fastshapelets">             
                /* Problem sets used in @article{rakthanmanon2013fast,
  title={Fast Shapelets: A Scalable Algorithm for Discovering Time Series Shapelets},
  author={Rakthanmanon, T. and Keogh, E.},
  journal={Proceedings of the 13th {SIAM} International Conference on Data Mining},
  year={2013}
}
All included except Cricket. There are three criket problems and they are not 
* alligned, the class values in the test set dont match

*/
		public static String[] fastShapeletProblems={	
			"ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SonyAIBORobotSurface", // 20,601,70,2
			"Beef", // 30,30,470,5
			"GunPoint", // 50,150,150,2
			"TwoLeadECG", // 23,1139,82,2
                        "Adiac", // 390,391,176,37
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"Coffee", // 28,28,286,2
			"DiatomSizeReduction", // 16,306,345,4
			"ECGFiveDays", // 23,861,136,2
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fish", // 175,175,463,7
			"Lighting2", // 60,61,637,2
			"Lighting7", // 70,73,319,7
			"FaceAll", // 560,1690,131,14
			"MALLAT", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"wafer", // 1000,6164,152,2
                        "yoga",
                        "FaceAll",
                        "TwoPatterns",
        		"CinC_ECG_torso" // 40,1380,1639,4
                };
//</editor-fold>
  
   
    //<editor-fold defaultstate="collapsed" desc="marteau09stiffness: TWED">                 
  		public static String[] marteau09stiffness={
			"SyntheticControl", // 300,300,60,6
			"GunPoint", // 50,150,150,2
			"CBF", // 30,900,128,3
			"FaceAll", // 560,1690,131,14
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"fiftywords", // 450,455,270,50
			"Trace", // 100,100,275,4
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"FaceFour", // 24,88,350,4
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"ECG200", // 100,100,96,2
			"Adiac", // 390,391,176,37
			"yoga", // 300,3000,426,2
			"fish", // 175,175,463,7
			"Coffee", // 28,28,286,2
			"OliveOil", // 30,30,570,4
			"Beef" // 30,30,470,5
                };  
                //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="stefan13movesplit: Move-Split-Merge">                 
  		public static String[] stefan13movesplit={
			"Coffee", // 28,28,286,2
			"CBF", // 30,900,128,3
			"ECG200", // 100,100,96,2
			"SyntheticControl", // 300,300,60,6
			"GunPoint", // 50,150,150,2
			"FaceFour", // 24,88,350,4
			"Lightning7", // 70,73,319,7
			"Trace", // 100,100,275,4
			"Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
			"Lightning2", // 60,61,637,2
			"OliveOil", // 30,30,570,4
                        "OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"fish", // 175,175,463,7
                        "FaceAll", // 560,1690,131,14
			"fiftywords", // 450,455,270,50
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"yoga" // 300,3000,426,2
                };  
                //</editor-fold>

                
public static String createMatlabList(String[] names){
    String res="";
    for(String s:names)
        res+="'"+s+"',";
    return res;
}
public static void main(String[] args){
    String s=createMatlabList(fileNames);
    System.out.println(s);
}
}

