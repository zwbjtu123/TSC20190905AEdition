/*
First hacky go at reproducing RPCD from 
@inproceedings{silva13recurrence,
author="D. Silva, V. de Souza, G. Batista",
title="Time Series Classification Using Compression Distance of Recurrence Plots",
booktitle    ="Proc. {IEEE} ICDM",
year="2013"
}
 */
package other_peoples_algorithms;

/**
 *
 * @author ajb
 */
public class Silva_RPCD {
    
    public static double[][] RPMatrix(double [] x){
        double[][] S=new double[x.length][x.length];
        double max=0;
        for(int i=0;i<x.length;i++)
            for(int j=0;j<x.length;j++){
                S[i][j]=Math.abs(x[i]-x[j]);
                if(S[i][j]>max)
                    max=S[i][j];
            }
        
        for(int i=0;i<x.length;i++)
            for(int j=0;j<x.length;j++)
                S[i][j]/=max;
        
        return S;
    }
    public static double CK1Distance(double[][] X, double[][] Y){
//    function distance = CK1Distance(x, y)
//distance = (( mpegSize(x, y) + mpegSize(y, x))/…
// (mpegSize(x, x) + mpegSize(y, y)))−1

    
/* Matlab
    N = length(x);
	S = zeros(N, N);

	for i = 1 : N
        S(:,i) = abs( repmat( x(i), N, 1 ) - x(:) );
    end
	% Normalize, in order to use imwrite correctly
    S = S./max(max(S));
end
*/    
        return 0;
    }
}
