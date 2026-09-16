#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int stride = 1 << (d + 1);
            int k = index * stride;

            if (k + stride - 1 < n) {
                int iPrev = k + (1 << d) - 1;
                // x[k + 2^(d+1) - 1] += x[k + 2^d - 1]
                data[k + stride - 1] += data[iPrev];   
            }
        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int stride = 1 << (d + 1);
            int k = index * stride;

            if (k + stride - 1 < n) {
                int left = k + (1 << d) - 1;
                int right = k + stride - 1;
                int t = data[left];
                data[left] = data[right];
                data[right] += t;
            }
        }

        void scanOnDevice(int n, int* dev_data) {
            int n2 = 1 << ilog2ceil(n);// power of 2
            // upsweep 
            for (int d = 0; d < ilog2ceil(n); ++d) {
                int stride = 1 << (d + 1);
                int numActive = n2 / stride;
                if (numActive == 0) continue; // no active elements at this level - skip launch
                int gridSize = (numActive + BLOCKSIZE - 1) / BLOCKSIZE;

                kernUpSweep << <gridSize, BLOCKSIZE >> > (n2, d, dev_data);
                checkCUDAError("kernUpSweep failed");
            }
            // downsweep 
            // rightmost element to 0
            cudaMemset(dev_data + n2 - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_data[n2-1] failed");
            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                int stride = 1 << (d + 1);
                int numActive = n2 / stride;
                if (numActive == 0) continue; // no active elements at this level - skip launch
                int gridSize = (numActive + BLOCKSIZE - 1) / BLOCKSIZE;

                kernDownSweep << <gridSize, BLOCKSIZE >> > (n2, d, dev_data);
                checkCUDAError("kernDownSweep failed");
            }
        }

        void scan(int n, int *odata, const int *idata) {
            int n2 = 1 << ilog2ceil(n);// power of 2

            int* dev_data;
            cudaMalloc((void**)&dev_data, n2 * sizeof(int));
            cudaMemset(dev_data, 0, n2 * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            scanOnDevice(n2, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);  
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int n2 = 1 << ilog2ceil(n);// power of 2

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
			int* dev_indices; 

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMalloc((void**)&dev_indices, n2 * sizeof(int)); // must be power of 2 length for scan
            checkCUDAError("compact: cudaMalloc failed");

            // cpu to gpu 
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("compact: memcpy dev_idata failed");

            int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;

            timer().startGpuTimer();
            // 1. temporary boolean array
            StreamCompaction::Common::kernMapToBoolean << <gridSize, BLOCKSIZE >> > (n, dev_bools, dev_idata);
            checkCUDAError("compact: kernMapToBoolean failed");
			// 2. scan boolean array
            cudaMemset(dev_indices, 0, n2 * sizeof(int));
            checkCUDAError("compact: cudaMemset dev_indices failed");
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice); // copy over bools
            checkCUDAError("compact: memcpy dev_bools->dev_indices failed");
            scanOnDevice(n2, dev_indices);
            // 3. scatter
            StreamCompaction::Common::kernScatter << <gridSize, BLOCKSIZE >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("compact: kernScatter failed");
            timer().endGpuTimer();

            // gpu to cpu 
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("compact: memcpy lastBool/lastIndex failed");
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("compact: memcpy odata failed");
            cudaFree(dev_idata);
            cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
            return count;
        }
    }
}
