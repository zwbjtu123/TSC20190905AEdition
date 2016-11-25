package development;

import fileIO.InFile;
import fileIO.OutFile;
import statistics.tests.OneSampleTests;
import statistics.tests.TwoSampleTests;

/**
 * Reads in a file of accuracies for k classifiers and generates a kxk matrix
 * of p-values. INPUT FORMAT:
 *          ,Classifier1,Classifier2,Classifier3, ...,Classifierk
 * Problem1 , 0.5,....
 * Problem2 , 0.5,....
 * .
 * .
 * ProblemN, 0.5,....
 * 
 * Output: Pairwise matrix of difference and a version of cliques.
 * 
 * @author ajb
 */
public class MultipleClassifiersPairwiseTest {
    static double[][] accs; //ROW indicates classifier, for ease of processing
    static double[][] pValsTTest; //ROW indicates classifier, for ease of processing
    static double[][] pValsSignTest; //ROW indicates classifier, for ease of processing
    static double[][] pValsSignRankTest; //ROW indicates classifier, for ease of processing
    static double[][] bonferonni_pVals; //ROW indicates classifier, for ease of processing
    static boolean[][] noDifference; //ROW indicates classifier, for ease of processing
    
    static int nosClassifiers;
    static int nosProblems;
    static String[] names;
    
    public static void loadData(String file){
        InFile data=new InFile(file);
        nosProblems=data.countLines()-1;
        data=new InFile(file);
        String[] temp=data.readLine().split(",");
        nosClassifiers=temp.length-1;
        names=new String[nosClassifiers];
        for(int i=0;i<nosClassifiers;i++)
            names[i]=temp[i+1];
        accs=new double[nosClassifiers][nosProblems];
        for(int j=0;j<nosProblems;j++){
            String t=data.readString(); //Problem name
            System.out.print("Problem ="+t+",");
            for(int i=0;i<nosClassifiers;i++){
                accs[i][j]=data.readDouble();
                System.out.print(accs[i][j]+",");
            }
                System.out.print("\n");
            
        }
    }
    
    public static void findPVals(){
        pValsTTest=new double[nosClassifiers][nosClassifiers];
        pValsSignTest=new double[nosClassifiers][nosClassifiers];
        pValsSignRankTest=new double[nosClassifiers][nosClassifiers];
        OneSampleTests test=new OneSampleTests();
        for(int i=0;i<nosClassifiers;i++)
        {
            for(int j=i+1;j<nosClassifiers;j++){
//Find differences
                double[] diff=new double[accs[i].length];
                for(int k=0;k<accs[i].length;k++)
                    diff[k]=accs[i][k]-accs[j][k];
                String str=test.performTests(diff);
                System.out.println("TEST Classifier "+names[i]+" VS "+names[j]+" IS "+str);
                String[] tmp=str.split(",");
                pValsTTest[i][j]=Double.parseDouble(tmp[2]);
                pValsSignTest[i][j]=Double.parseDouble(tmp[5]);
                pValsSignRankTest[i][j]=Double.parseDouble(tmp[8]);
            }
        }
    }
    
    public static void findDifferences(double alpha,boolean printPVals){
        noDifference=new boolean[nosClassifiers][nosClassifiers];
        for(int i=0;i<nosClassifiers;i++)
        {
            noDifference[i][i]=true;
            for(int j=i+1;j<nosClassifiers;j++){
                noDifference[i][j]=true;
                noDifference[j][i]=true;
                if(pValsSignRankTest[i][j]<alpha){
//                if(pValsTTest[i][j]<alpha && pValsSignTest[i][j]< alpha && pValsSignRankTest[i][j]<alpha)
                    noDifference[i][j]=false;
                    noDifference[j][i]=false;
                }
            }
        }
        
        System.out.print("\t");
        for(int i=0;i<nosClassifiers;i++)
            System.out.print(names[i]+"\t");
        System.out.println(" alpha =\n"+alpha);
        for(int i=0;i<nosClassifiers;i++){
            System.out.print(names[i]+"\t");
            for(int j=0;j<nosClassifiers;j++){
                if(j<=i)
                    System.out.print("\t");
                else
                    if(printPVals)
                        System.out.print(pValsSignRankTest[i][j]+"\t");
                else
                    System.out.print(noDifference[i][j]+"\t");
            }
            System.out.println("\n");
        }
        
    } 
    public static void findCliques(){

        
        
    }
    
    public static void runTests(String input, String output){
         loadData(input);
//        loadData("C:\\Research\\Papers\\2016\\JMLR HIVE-COTE Jason\\RiseTestWithNames.csv");
        findPVals();
        double alpha=0.05;
//printPVals=false;
//Bonferonni adjusted        
//        alpha/=nosClassifiers*(nosClassifiers-1)/2;
//Control adjusted 
        alpha/=nosClassifiers-1;
        findDifferences(alpha,true);
        //Sort classifiers by rank: assume already done
        OutFile cliques=new OutFile(output);
        for(int i=0;i<nosClassifiers;i++){
            for(int j=0;j<nosClassifiers;j++)
                cliques.writeString(noDifference[i][j]+",");
            cliques.writeString("\n");
        }
       
    }
    public static void main(String[] args) {
//ASSUME INPUT IN RANK ORDER, WITH TOP RANKED CLASSIFIER FIRST, WORST LAST        
//        String input="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\BasicExperiments\\";
//        String output="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\BasicExperiments\\";
        String[] allSimulators={"WholeSeriesElastic","Interval","Shapelet","Dictionary","ARMA","All"};
//        for(String s:allSimulators)
        String input="C:\\Research\\Results\\RepoResults\\HIVE Results";
//        String s= "All";
//            runTests(input+s+"CombinedResults.csv",input+s+"Tests.csv");
            runTests(input+".csv",input+"Tests.csv");
    }
    
}
