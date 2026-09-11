#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (exclusive prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            if (n == 0) return;
            
            int currSum = 0;
			odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                 currSum += idata[i - 1];
				 odata[i] = currSum;
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int oindex = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
					odata[oindex] = idata[i];
                    oindex++;
                }
            }
            timer().endCpuTimer();
            return oindex;  
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

                // 1. compute temp array 
                int* temp = new int[n];
                for (int i = 0; i < n; ++i) {
					temp[i] = (idata[i] != 0) ? 1 : 0;
				}

				// 2. compute exclusive scan of temp array
				int* scanned = new int[n];
                int currSum = 0;
                scanned[0] = 0;
                for (int i = 1; i < n; ++i) {
                    currSum += temp[i - 1];
                    scanned[i] = currSum;
                }

                int oindex;
                int count = 0;
                // 3. scatter 
                for (int i = 0; i < n; i++) {
                    oindex = scanned[i];
                    if (temp[i] == 1) {
                        odata[oindex] = idata[i];
                        count++;
                    }
                }
            timer().endCpuTimer();
            return count;
        }
    }
}
