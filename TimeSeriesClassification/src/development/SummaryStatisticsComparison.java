/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import static development.NearestNeighbourTSC.path;
import fileIO.OutFile;
import java.util.ArrayList;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.filters.timeseries.SummaryStats;

/**
 *
 * @author ajb
 */
public class SummaryStatisticsComparison {

    public static void trainTestVector(){
        OutFile of=new OutFile(path+"Vector.csv");
        for(String s:DataSets.fileNames){
            try{
                vectorClassifiersSingleSample(s,of);
            }catch(Exception e){
                System.out.println(" Failed on problem : "+s+ "with exception "+e);
                e.printStackTrace();
                of.writeString("\n");
            }
        }
    }
    public static void trainTestSummary(){
        OutFile of=new OutFile(path+"SummaryStats.csv");
        for(String s:DataSets.fileNames){
            try{
                summaryStatsClassifiersSingleSample(s,of);
            }catch(Exception e){
                System.out.println(" Failed on problem : "+s+ "with exception "+e);
                e.printStackTrace();
                of.writeString("\n");
            }
        }
    }


       
    public static void summaryStatsClassifiersSingleSample(String fileName, OutFile results) throws Exception{
        Instances test=utilities.ClassifierTools.loadData(path+fileName+"\\"+fileName+"_TEST");
        Instances train=utilities.ClassifierTools.loadData(path+fileName+"\\"+fileName+"_TRAIN");
        results.writeString(fileName+",");
        System.out.print(fileName+",");
/*        Instances all=new Instances(train);
        for(Instance ins:test)
            all.add(ins);
        int testSize=test.numInstances();
    //Form new Train/Test split
        all.randomize(new Random());
        Instances tr = new Instances(all);
        Instances te= new Instances(all,0);
        for(int j=0;j<testSize;j++)
            te.add(tr.remove(0));            
*/
        SummaryStats ss = new SummaryStats();
        train=ss.process(train);
        test=ss.process(test);
        ArrayList<String> names=new ArrayList<>();
        Classifier[] c=ClassifierTools.setDefaultSingleClassifiers(names); 
        double acc;
        for(Classifier cl:c){
            acc=ClassifierTools.singleTrainTestSplitAccuracy(cl, train, test);
            results.writeString((1-acc)+",");
           System.out.print((1-acc)+",");
        }
        WeightedEnsemble we= new WeightedEnsemble();
        acc=ClassifierTools.singleTrainTestSplitAccuracy(we, train, test);
        results.writeLine((1-acc)+"");
        System.out.print((1-acc)+"\n");
    }
 
    public static void vectorClassifiersSingleSample(String fileName, OutFile results) throws Exception{
        Instances test=utilities.ClassifierTools.loadData(path+fileName+"\\"+fileName+"_TEST");
        Instances train=utilities.ClassifierTools.loadData(path+fileName+"\\"+fileName+"_TRAIN");
        results.writeString(fileName+",");
        System.out.print(fileName+",");
/*        Instances all=new Instances(train);
        for(Instance ins:test)
            all.add(ins);
        int testSize=test.numInstances();
    //Form new Train/Test split
        all.randomize(new Random());
        Instances tr = new Instances(all);
        Instances te= new Instances(all,0);
        for(int j=0;j<testSize;j++)
            te.add(tr.remove(0));            
*/
        ArrayList<String> names=new ArrayList<>();
        Classifier[] c=ClassifierTools.setDefaultSingleClassifiers(names); 
        double acc;
        for(Classifier cl:c){
            acc=ClassifierTools.singleTrainTestSplitAccuracy(cl, train, test);
            results.writeString((1-acc)+",");
           System.out.print((1-acc)+",");
        }
        WeightedEnsemble we= new WeightedEnsemble();
        acc=ClassifierTools.singleTrainTestSplitAccuracy(we, train, test);
        results.writeLine((1-acc)+"");
        System.out.print((1-acc)+"\n");
    }
 
    

}
