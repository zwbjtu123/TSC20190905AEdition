/*
This hacky version assumes that each dimension is concatenated into a single Instance.

So, if the multidimensional data is in three dimensions, say (x,y,z),
instances is 
x1,x2,x3,...xk,y1,y2, etc.

must pass the number of dimensions on construction. If the data is misalligned it will fail miserably


*/
package classifiers.lazy;

import weka.core.elastic_distance_measures.*;

/**
 *
 * @author ajb
 */
public class DTW_D extends DTW_DistanceBasic{
    int dimension;  //This 
    
    private DTW_D(){}   //Cannot construct without passing dimensions 
    public DTW_D(int d){
        dimension=d;
        System.out.println("Dimension ="+d);
    }   //Cannot construct without passing dimensions 
  
    private double pointWiseDistance(double[] a,double[] b, int t1, int t2){
//X dimension
        double dist=(a[t1]-b[t2])*(a[t1]-b[t2]);
        for(int i=1;i<dimension;i++)
            dist+=(a[i*dimension+t1]-b[i*dimension+t2])*(a[i*dimension+t1]-b[i*dimension+t2]);
        return dist;
    }
   @Override
    public final double distance(double[] a,double[] b, double cutoff){
        double minDist;
        boolean tooBig;
        int n=a.length/dimension;
        int m=n;
        if(a.length!=b.length || a.length%dimension!=0){     //Do not allow misalligned   
            System.out.println("ERROR, MISALLIGNED DATA");
            throw new RuntimeException("\"ERROR, MISALLIGNED DATA in DTW_D distance: a.length= "+a.length+"b.length= "+b.length);
        }
/*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp 
generalised for variable window size
* */
        windowSize = getWindowSize(n);
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV 
//for varying window sizes        
        if(matrixD==null)
            matrixD=new double[n][m];
        
/*
//Set boundary elements to max. 
*/
        int start,end;
        for(int i=0;i<n;i++){
            start=windowSize<i?i-windowSize:0;
            end=i+windowSize+1<m?i+windowSize+1:m;
            for(int j=start;j<end;j++)
                matrixD[i][j]=Double.MAX_VALUE;
        }
        matrixD[0][0]=pointWiseDistance(a,b,0,0);
//a is the longer series. 
//Base cases for warping 0 to all with max interval	r	
//Warp a[0] onto all b[1]...b[r+1]
        for(int j=1;j<windowSize && j<m;j++)
                matrixD[0][j]=matrixD[0][j-1]+pointWiseDistance(a,b,0,j);

//	Warp b[0] onto all a[1]...a[r+1]
        for(int i=1;i<windowSize && i<n;i++)
                matrixD[i][0]=matrixD[i-1][0]+pointWiseDistance(a,b,i,0);
//Warp the rest,
        for (int i=1;i<n;i++){
            tooBig=true; 
            start=windowSize<i?i-windowSize+1:1;
            end=i+windowSize<m?i+windowSize:m;
            for (int j = start;j<end;j++){
                    minDist=matrixD[i][j-1];
                    if(matrixD[i-1][j]<minDist)
                            minDist=matrixD[i-1][j];
                    if(matrixD[i-1][j-1]<minDist)
                            minDist=matrixD[i-1][j-1];
                    matrixD[i][j]=minDist+ pointWiseDistance(a,b,i,j);//         (a[i]-b[j])*(a[i]-b[j]);
                    if(tooBig&&matrixD[i][j]<cutoff)
                            tooBig=false;               
            }
            //Early abandon
            if(tooBig){
                return Double.MAX_VALUE;
            }
        }			
//Find the minimum distance at the end points, within the warping window. 
        return matrixD[n-1][m-1];
    }
     
    
    
    
}
