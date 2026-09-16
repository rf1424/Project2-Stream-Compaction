#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int offset, int* odata, const int* idata) {

            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) return;

            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {


            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            // cpu to gpu 
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;

            timer().startGpuTimer();

            for (int i = 1; i < n; i *= 2) {
                kernNaiveScan << <gridSize, BLOCKSIZE >> > (n, i, dev_odata, dev_idata);

                // swap
                int* temp = dev_odata;
                dev_odata = dev_idata;
                dev_idata = temp;
            }

            // inclusive scan -> exclusive scan 
            cudaMemset(dev_odata, 0, sizeof(int));
            cudaMemcpy(dev_odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToDevice);

            timer().endGpuTimer();

            // gpu to cpu
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}