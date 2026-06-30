/*
 CSC580 Parallel Computing
 Corrected MPI Statistical Data Analytics Pipeline

 Compile in x64 Native Tools Command Prompt:
 cl /EHsc /O2 mpi_analytics.cpp /I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /link "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib" /out:mpi_analytics.exe

 Run:
 mpiexec -n 4 mpi_analytics.exe 1000000
*/

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;

vector<double> generateData(long long N) {
    vector<double> data(N);
    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0, 10000.0);

    for (long long i = 0; i < N; i++) {
        data[i] = dist(rng);
    }

    return data;
}

vector<double> generateSecondColumn(const vector<double>& data) {
    vector<double> data2(data.size());
    mt19937_64 rng(99);
    uniform_real_distribution<double> noise(0.0, 10000.0);

    for (size_t i = 0; i < data.size(); i++) {
        data2[i] = (0.75 * data[i]) + (0.25 * noise(rng));
    }

    return data2;
}

void saveTimingToCsv(const string& filename,
                     long long datasetSize,
                     int processes,
                     double totalTime,
                     double statTime,
                     double histogramTime,
                     double sortingTime,
                     double correlationTime,
                     double movingAverageTime,
                     double outlierTime) {
    bool fileExists = false;

    ifstream check(filename);
    if (check.good()) fileExists = true;
    check.close();

    ofstream file(filename, ios::app);

    if (!fileExists) {
        file << "DatasetSize,Processes,Total_ms,BasicStatistics_ms,Histogram_ms,Sorting_ms,"
             << "PearsonCorrelation_ms,MovingAverage_ms,OutlierDetection_ms\n";
    }

    file << fixed << setprecision(4)
         << datasetSize << ","
         << processes << ","
         << totalTime << ","
         << statTime << ","
         << histogramTime << ","
         << sortingTime << ","
         << correlationTime << ","
         << movingAverageTime << ","
         << outlierTime << "\n";

    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 1000000;

    if (rank == 0) {
        if (argc > 1) N = atoll(argv[1]);

        cout << "========================================\n";
        cout << " MPI Statistical Analytics Program\n";
        cout << "========================================\n";
        cout << "Dataset size        : " << N << "\n";
        cout << "Number of processes : " << size << "\n";
    }

    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    long long base = N / size;
    long long remainder = N % size;

    vector<int> sendCounts(size);
    vector<int> displacements(size);

    long long offset = 0;
    for (int i = 0; i < size; i++) {
        long long chunk = base + (i < remainder ? 1 : 0);
        sendCounts[i] = static_cast<int>(chunk);
        displacements[i] = static_cast<int>(offset);
        offset += chunk;
    }

    int localCount = sendCounts[rank];

    vector<double> fullData;
    vector<double> fullData2;

    double totalStart = MPI_Wtime();

    if (rank == 0) {
        fullData = generateData(N);
        fullData2 = generateSecondColumn(fullData);
    }

    vector<double> localData(localCount);
    vector<double> localData2(localCount);

    MPI_Scatterv(rank == 0 ? fullData.data() : nullptr,
                 sendCounts.data(), displacements.data(), MPI_DOUBLE,
                 localData.data(), localCount, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? fullData2.data() : nullptr,
                 sendCounts.data(), displacements.data(), MPI_DOUBLE,
                 localData2.data(), localCount, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Keep original unsorted copy for correlation, moving average, outlier
    vector<double> originalLocalData = localData;

    MPI_Barrier(MPI_COMM_WORLD);

    // =====================================================
    // 1. BASIC STATISTICS
    // =====================================================
    double statStart = MPI_Wtime();

    double localSum = 0.0;
    double localMin = originalLocalData[0];
    double localMax = originalLocalData[0];

    for (double x : originalLocalData) {
        localSum += x;
        if (x < localMin) localMin = x;
        if (x > localMax) localMax = x;
    }

    double globalSum = 0.0;
    double globalMin = 0.0;
    double globalMax = 0.0;

    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double mean = 0.0;
    if (rank == 0) mean = globalSum / N;

    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double localVarianceSum = 0.0;
    for (double x : originalLocalData) {
        localVarianceSum += (x - mean) * (x - mean);
    }

    double globalVarianceSum = 0.0;
    MPI_Reduce(&localVarianceSum, &globalVarianceSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double variance = 0.0;
    double stddev = 0.0;

    if (rank == 0) {
        variance = globalVarianceSum / N;
        stddev = sqrt(variance);
    }

    MPI_Bcast(&variance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double statEnd = MPI_Wtime();
    double statTime = (statEnd - statStart) * 1000.0;

    // =====================================================
    // 2. HISTOGRAM
    // =====================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double histStart = MPI_Wtime();

    const int bins = 10;
    int localHist[bins] = {0};
    int globalHist[bins] = {0};

    for (double x : originalLocalData) {
        int bin = static_cast<int>(x / 1000.0);
        if (bin < 0) bin = 0;
        if (bin >= bins) bin = bins - 1;
        localHist[bin]++;
    }

    MPI_Reduce(localHist, globalHist, bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double histEnd = MPI_Wtime();
    double histogramTime = (histEnd - histStart) * 1000.0;

    // =====================================================
    // 3. SORTING
    // =====================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double sortStart = MPI_Wtime();

    vector<double> sortedLocalData = originalLocalData;
    sort(sortedLocalData.begin(), sortedLocalData.end());

    vector<double> gatheredData;
    if (rank == 0) gatheredData.resize(N);

    MPI_Gatherv(sortedLocalData.data(), localCount, MPI_DOUBLE,
                rank == 0 ? gatheredData.data() : nullptr,
                sendCounts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double smallest = 0.0;
    double largest = 0.0;
    double median = 0.0;

    if (rank == 0) {
        sort(gatheredData.begin(), gatheredData.end());
        smallest = gatheredData.front();
        largest = gatheredData.back();
        median = gatheredData[gatheredData.size() / 2];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double sortEnd = MPI_Wtime();
    double sortingTime = (sortEnd - sortStart) * 1000.0;

    // =====================================================
    // 4. PEARSON CORRELATION
    // =====================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double corrStart = MPI_Wtime();

    double localSumX = 0.0;
    double localSumY = 0.0;
    double localSumXY = 0.0;
    double localSumX2 = 0.0;
    double localSumY2 = 0.0;

    for (int i = 0; i < localCount; i++) {
        localSumX += originalLocalData[i];
        localSumY += localData2[i];
        localSumXY += originalLocalData[i] * localData2[i];
        localSumX2 += originalLocalData[i] * originalLocalData[i];
        localSumY2 += localData2[i] * localData2[i];
    }

    double globalSumX = 0.0;
    double globalSumY = 0.0;
    double globalSumXY = 0.0;
    double globalSumX2 = 0.0;
    double globalSumY2 = 0.0;

    MPI_Reduce(&localSumX, &globalSumX, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localSumY, &globalSumY, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localSumXY, &globalSumXY, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localSumX2, &globalSumX2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localSumY2, &globalSumY2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double correlation = 0.0;

    if (rank == 0) {
        double numerator = (N * globalSumXY) - (globalSumX * globalSumY);
        double denominator = sqrt(((N * globalSumX2) - (globalSumX * globalSumX)) *
                                  ((N * globalSumY2) - (globalSumY * globalSumY)));

        if (denominator != 0.0) correlation = numerator / denominator;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double corrEnd = MPI_Wtime();
    double correlationTime = (corrEnd - corrStart) * 1000.0;

    // =====================================================
    // 5. MOVING AVERAGE
    // =====================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double movingStart = MPI_Wtime();

    const int window = 5;

    double firstMovingAverage = 0.0;
    double lastMovingAverage = 0.0;

    if (rank == 0 && fullData.size() >= window) {
        double firstSum = 0.0;
        double lastSum = 0.0;

        for (int i = 0; i < window; i++) firstSum += fullData[i];
        for (long long i = N - window; i < N; i++) lastSum += fullData[i];

        firstMovingAverage = firstSum / window;
        lastMovingAverage = lastSum / window;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double movingEnd = MPI_Wtime();
    double movingAverageTime = (movingEnd - movingStart) * 1000.0;

    // =====================================================
    // 6. OUTLIER DETECTION
    // =====================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double outlierStart = MPI_Wtime();

    double lowerBound = mean - (3.0 * stddev);
    double upperBound = mean + (3.0 * stddev);

    long long localOutliers = 0;

    for (double x : originalLocalData) {
        if (x < lowerBound || x > upperBound) localOutliers++;
    }

    long long globalOutliers = 0;
    MPI_Reduce(&localOutliers, &globalOutliers, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double outlierEnd = MPI_Wtime();
    double outlierTime = (outlierEnd - outlierStart) * 1000.0;

    double totalEnd = MPI_Wtime();
    double totalTime = (totalEnd - totalStart) * 1000.0;

    if (rank == 0) {
        cout << fixed << setprecision(4);

        cout << "\n[1] MPI Basic Statistics\n";
        cout << "Mean               : " << mean << "\n";
        cout << "Variance           : " << variance << "\n";
        cout << "Standard Deviation : " << stddev << "\n";
        cout << "Minimum            : " << globalMin << "\n";
        cout << "Maximum            : " << globalMax << "\n";

        cout << "\n[2] MPI Histogram\n";
        for (int i = 0; i < bins; i++) {
            cout << setw(5) << i * 1000 << " - " << setw(5) << ((i + 1) * 1000 - 1)
                 << " : " << globalHist[i] << "\n";
        }

        cout << "\n[3] MPI Sorting\n";
        cout << "Smallest value     : " << smallest << "\n";
        cout << "Largest value      : " << largest << "\n";
        cout << "Median value       : " << median << "\n";

        cout << "\n[4] MPI Pearson Correlation\n";
        cout << "Correlation value  : " << correlation << "\n";

        cout << "\n[5] MPI Moving Average\n";
        cout << "First moving avg   : " << firstMovingAverage << "\n";
        cout << "Last moving avg    : " << lastMovingAverage << "\n";

        cout << "\n[6] MPI Outlier Detection\n";
        cout << "Outlier count      : " << globalOutliers << "\n";

        cout << "\n========================================\n";
        cout << " MPI Timing Result\n";
        cout << "========================================\n";
        cout << "Basic Statistics      : " << statTime << " ms\n";
        cout << "Histogram             : " << histogramTime << " ms\n";
        cout << "Sorting               : " << sortingTime << " ms\n";
        cout << "Pearson Correlation   : " << correlationTime << " ms\n";
        cout << "Moving Average        : " << movingAverageTime << " ms\n";
        cout << "Outlier Detection     : " << outlierTime << " ms\n";
        cout << "Total Runtime         : " << totalTime << " ms\n";
        cout << "========================================\n";

        saveTimingToCsv("../results/mpi_results.csv", N, size, totalTime,
                        statTime, histogramTime, sortingTime,
                        correlationTime, movingAverageTime, outlierTime);

        cout << "\nTiming saved to ../results/mpi_results.csv\n";
    }

    MPI_Finalize();
    return 0;
}
