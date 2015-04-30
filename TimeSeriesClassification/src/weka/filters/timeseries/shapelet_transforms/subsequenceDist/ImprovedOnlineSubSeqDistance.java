/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.subsequenceDist;

/**
 *
 * @author raj09hxu
 */
public class ImprovedOnlineSubSeqDistance extends OnlineSubSeqDistance{

    @Override
    public double calculate(double[] timeSeries, int startPos)
    {
        DoubleWrapper sumPointer = new DoubleWrapper();
        DoubleWrapper sum2Pointer = new DoubleWrapper();

        //Generate initial subsequence that starts at the same position our candidate does.
        double[] subseq = new double[candidate.length];
        System.arraycopy(timeSeries, startPos, subseq, 0, subseq.length);
        subseq = optimizedZNormalise(subseq, false, sumPointer, sum2Pointer);
        
        double bestDist = 0.0;
        double temp;
        
        //Compute initial distance. from the startPosition the candidate was found.
        for (int i = 0; i < candidate.length; i++)
        {
            temp = candidate[i] - subseq[i];
            bestDist = bestDist + (temp * temp);
        }

        int i=1;
        double currentDist;
        
        int[] pos = new int[2];
        double[] sum = {sumPointer.get(), sumPointer.get()};
        double[] sumsq = {sum2Pointer.get(), sum2Pointer.get()};
        boolean[] traverse = new boolean[2];
        
        while(traverse[0] || traverse[1])
        {
            //i will be 0 and 1.
            for(int j=0; j<2; j++)
            {
                int modifier = i==0 ? -1 : 1;
                
                pos[i] = startPos + modifier; 
                
                //if we're going left check we're greater than 0 if we're going right check we've got room to move.
                traverse[i] = j==0 ? pos[i] >=0 : pos[i] < timeSeries.length - candidate.length;
                
                if(!traverse[i])
                    continue;
                
                //either take off nothing, or take off 1. This gives us our offset.
                double start = timeSeries[pos[i]-j];
                double end   = timeSeries[pos[i]-j + candidate.length];
                
                // we want to invert the modifier. IE the right wants to be *-1 and the left wants to just be 1.
                sum[i] = sum[i] + ((modifier*-1)*(start + end));
                sumsq[i] = sum[i] + ((modifier*-1)*((start * start) + (end * end)));

                currentDist = calculateBestDistance(pos[i], timeSeries, candidate, sortedIndices, bestDist, sum[i], sumsq[i]);  
                
                if (currentDist < bestDist)
                {
                    bestDist = currentDist;
                }
            }
            i++;
        }

        return (bestDist == 0.0) ? 0.0 : (1.0 / candidate.length * bestDist);
    }

    
}
