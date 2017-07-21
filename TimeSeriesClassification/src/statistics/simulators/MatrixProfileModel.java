/*
AJB Oct 2016

Model to simulate data where matrix profile should be optimal.

Two locations are common to a class. The sample random shape (with random AMP and
BASE) are placed in the same class for each series of this model.

With Generate: 
if in first shape, randomly generate. If in 


*/
package statistics.simulators;
import fileIO.OutFile;
import java.util.*;
import java.io.*;
import statistics.distributions.NormalDistribution;
import statistics.simulators.DictionaryModel.ShapeType;
import static statistics.simulators.Model.rand;
import statistics.simulators.ShapeletModel.Shape;

public class MatrixProfileModel extends Model {
    private int nosLocations=2; //
    private int shapeLength=29;
    public static double MINBASE=-2;
    public static double MINAMP=.1;
    public static double MAXBASE=2;
    public static double MAXAMP=2;
    DictionaryModel.Shape shape;//Will change for each series
    private static int GLOBALSERIESLENGTH=500;
    private int seriesLength; // Need to set intervals, maybe allow different lengths? 
    private int base=-1;
    private int amplitude=2;
    private int shapeCount=0;
    ArrayList<Integer>  locations;
    public MatrixProfileModel(){
        shapeCount=rand.nextInt(ShapeType.values().length);
        seriesLength=GLOBALSERIESLENGTH;
        locations=new ArrayList<>();    
        setNonOverlappingIntervals();
        
    }
    public void setSeriesLength(int n){
        seriesLength=n;
    }
    public static void setGlobalSeriesLength(int n){
        GLOBALSERIESLENGTH=n;
    }
   public void setNonOverlappingIntervals(){
//Me giving up and just randomly placing the shapes until they are all non overlapping
        for(int i=0;i<nosLocations;i++){
            boolean ok=false;
            int l=shapeLength/2;
            while(!ok){
                ok=true;
//Search mid points to level the distribution up somewhat
//                System.out.println("Series length ="+seriesLength);
                l=rand.nextInt(seriesLength-shapeLength)+shapeLength/2;
//          System.out.println("trying   "+l);
                
                for(int in:locations){
//I think this is setting them too big                    
                    if((l>=in-shapeLength && l<in+shapeLength) //l inside ins
                      ||(l<in-shapeLength && l+shapeLength>in)      ){ //ins inside l
                        ok=false;
//                       System.out.println(l+"  overlaps with "+in);
                        break;
                    }
                }
            }
//           System.out.println("Adding "+l);
            locations.add(l);
        }
//Revert to start points            
        for(int i=0;i<locations.size();i++){
            int val=locations.get(i);
            locations.set(i, val-shapeLength/2);
        }
        Collections.sort(locations);
    }

    @Override
    public void setParameters(double[] p) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    public void setLocations(ArrayList<Integer> l, int length){
        locations=new ArrayList<>(l);    
        shapeLength=length;
    }
    public ArrayList<Integer> getIntervals(){return locations;}
    public int getShapeLength(){ return shapeLength;}
    
    public void generateBaseShape(){
//Randomise BASE and AMPLITUDE
        double b=MINBASE+(MAXBASE-MINBASE)*Model.rand.nextDouble();
        double a=MINAMP+(MAXAMP-MINAMP)*Model.rand.nextDouble();
        ShapeType[] all=ShapeType.values();
            ShapeType st=all[(shapeCount++)%all.length];
            shape=new DictionaryModel.Shape(st,shapeLength,b,a);
 //           System.out.println("Shape is "+shape);
 //        shape.nextShape();
//        shape
    }
    
     @Override   
	public	double[] generateSeries(int n)
	{
           t=0;
//Resets the starting locations each time this is called          
           generateBaseShape();
           double[] d = new double[n];
           for(int i=0;i<n;i++)
              d[i]=generate();
           return d;
        }
    
    
    @Override
    public double generate(){
//Noise
//        System.out.println("Error var ="+error.getVariance());
        double value=error.simulate();
        int insertionPoint=0;
        while(insertionPoint<locations.size() && locations.get(insertionPoint)+shapeLength<t)
            insertionPoint++;    
        if(insertionPoint>=locations.size()){ //Bigger than all the start points, set to last
            insertionPoint=locations.size()-1;
        }
        int point=locations.get(insertionPoint);
        if(point<=t && point+shapeLength>t){//in shape1
            value+=shape.generateWithinShapelet((int)(t-point));
//                System.out.println(" IN SHAPE 1 occurence "+insertionPoint+" Time "+t);
        }
        t++;
        return value;
    }
    
    public static void generateExampleData(){
        int length=500;
        GLOBALSERIESLENGTH=length;
        Model.setGlobalRandomSeed(3);
        Model.setDefaultSigma(1);
        MatrixProfileModel m1=new MatrixProfileModel();
        MatrixProfileModel m2=new MatrixProfileModel();
        
        double[][] d=new double[10][];
        for(int i=0;i<5;i++){
            d[i]=m1.generateSeries(length);
        }
        for(int i=5;i<10;i++){
            d[i]=m2.generateSeries(length);
        }
        OutFile of=new OutFile("C:\\temp\\MP_ExampleSeriesNoise.csv");
        for(int i=0;i<length;i++){
            for(int j=0;j<10;j++)
                of.writeString(d[j][i]+",");
            of.writeString("\n");
        }
        
    }
    public static void main(String[] args){
        generateExampleData();
        System.exit(0);

//Set up two models with same intervals but different shapes        
        int length=500;
        Model.setGlobalRandomSeed(10);
        Model.setDefaultSigma(0.1);
        MatrixProfileModel m1=new MatrixProfileModel();
        MatrixProfileModel m2=new MatrixProfileModel();
        double[] d1=m1.generateSeries(length);
        double[] d2=m2.generateSeries(length);
        OutFile of=new OutFile("C:\\temp\\MP_Ex.csv");
        for(int i=0;i<length;i++)
            of.writeLine(d1[i]+","+d2[i]);
    }
    
}
