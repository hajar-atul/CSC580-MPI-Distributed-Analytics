/*
 CSC580 Parallel Computing
 Sequential Statistical Data Analytics Pipeline

 This sequential version is used as the baseline for comparison with MPI.
 It implements all required analytical tasks:
 1. Basic statistics
 2. Histogram generation
 3. Sorting
 4. Pearson correlation
 5. Moving average
 6. Outlier detection

 Compile:
 cl /EHsc /O2 sequential_analytics.cpp /Fe:sequential_analytics.exe

 Run:
 sequential_analytics.exe 1000000
 sequential_analytics.exe 10000000
 sequential_analytics.exe 100000000
*/

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;
using namespace chrono;

struct Statistics {
    double mean;
    double variance;
    double stddev;
    double minimum;
    double maximum;
};

struct TaskTiming {
    double generationTime;
    double statisticsTime;
    double histogramTime;
    double sortingTime;
    double correlationTime;
    double movingAverageTime;
    double outlierTime;
    double totalTime;
};

double elapsedMs(high_resolution_clock::time_point start,
                 high_resolution_clock::time_point end) {
    return duration<double, milli>(end - start).count();
}

vector<double> generateData(size_t N) {
    vector<double> data;
    data.reserve(N);

    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0, 10000.0);

    for (size_t i = 0; i < N; i++) {
        data.push_back(dist(rng));
    }

    return data;
}

vector<double> generateSecondColumn(const vector<double>& data) {
    vector<double> data2;
    data2.reserve(data.size());

    mt19937_64 rng(99);
    uniform_real_distribution<double> noise(0.0, 10000.0);

    for (double x : data) {
        data2.push_back((0.75 * x) + (0.25 * noise(rng)));
    }

    return data2;
}

Statistics calculateStatistics(const vector<double>& data) {
    Statistics stats;

    double sum = accumulate(data.begin(), data.end(), 0.0);
    stats.mean = sum / data.size();

    double varianceSum = 0.0;
    for (double value : data) {
        varianceSum += (value - stats.mean) * (value - stats.mean);
    }

    stats.variance = varianceSum / data.size();
    stats.stddev = sqrt(stats.variance);
    stats.minimum = *min_element(data.begin(), data.end());
    stats.maximum = *max_element(data.begin(), data.end());

    return stats;
}

vector<size_t> generateHistogram(const vector<double>& data, int binCount) {
    vector<size_t> histogram(binCount, 0);

    double minRange = 0.0;
    double maxRange = 10000.0;
    double binWidth = (maxRange - minRange) / binCount;

    for (double value : data) {
        int binIndex = static_cast<int>((value - minRange) / binWidth);

        if (binIndex < 0) {
            binIndex = 0;
        }

        if (binIndex >= binCount) {
            binIndex = binCount - 1;
        }

        histogram[binIndex]++;
    }

    return histogram;
}

vector<double> sortData(const vector<double>& data) {
    vector<double> sortedData = data;
    sort(sortedData.begin(), sortedData.end());
    return sortedData;
}

double calculatePearsonCorrelation(const vector<double>& x,
                                   const vector<double>& y) {
    size_t N = x.size();

    double sumX = 0.0;
    double sumY = 0.0;
    double sumXY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;

    for (size_t i = 0; i < N; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }

    double numerator = (N * sumXY) - (sumX * sumY);
    double denominator = sqrt(((N * sumX2) - (sumX * sumX)) *
                              ((N * sumY2) - (sumY * sumY)));

    if (denominator == 0.0) {
        return 0.0;
    }

    return numerator / denominator;
}

vector<double> calculateMovingAverage(const vector<double>& data, int windowSize) {
    vector<double> movingAverage;

    if (data.size() < static_cast<size_t>(windowSize)) {
        return movingAverage;
    }

    movingAverage.reserve(data.size() - windowSize + 1);

    double windowSum = 0.0;

    for (size_t i = 0; i < data.size(); i++) {
        windowSum += data[i];

        if (i >= static_cast<size_t>(windowSize)) {
            windowSum -= data[i - windowSize];
        }

        if (i >= static_cast<size_t>(windowSize - 1)) {
            movingAverage.push_back(windowSum / windowSize);
        }
    }

    return movingAverage;
}

size_t detectOutliers(const vector<double>& data,
                      double mean,
                      double stddev) {
    double lowerBound = mean - (3.0 * stddev);
    double upperBound = mean + (3.0 * stddev);

    size_t outlierCount = 0;

    for (double value : data) {
        if (value < lowerBound || value > upperBound) {
            outlierCount++;
        }
    }

    return outlierCount;
}

void saveTimingToCsv(const string& filename,
                     size_t datasetSize,
                     const TaskTiming& timing) {
    bool fileExists = false;

    ifstream checkFile(filename);
    if (checkFile.good()) {
        fileExists = true;
    }
    checkFile.close();

    ofstream file(filename, ios::app);

    if (!fileExists) {
        file << "DatasetSize,Generation_ms,BasicStatistics_ms,Histogram_ms,Sorting_ms,"
             << "PearsonCorrelation_ms,MovingAverage_ms,OutlierDetection_ms,Total_ms\n";
    }

    file << fixed << setprecision(4)
         << datasetSize << ","
         << timing.generationTime << ","
         << timing.statisticsTime << ","
         << timing.histogramTime << ","
         << timing.sortingTime << ","
         << timing.correlationTime << ","
         << timing.movingAverageTime << ","
         << timing.outlierTime << ","
         << timing.totalTime << "\n";

    file.close();
}

void printSummary(size_t datasetSize,
                  const Statistics& stats,
                  const vector<size_t>& histogram,
                  const vector<double>& sortedData,
                  double correlation,
                  const vector<double>& movingAverage,
                  size_t outlierCount,
                  const TaskTiming& timing) {
    cout << fixed << setprecision(4);

    cout << "\n========================================\n";
    cout << " Sequential Statistical Analytics Result\n";
    cout << "========================================\n";
    cout << "Dataset size: " << datasetSize << "\n";

    cout << "\n[1] Basic Statistics\n";
    cout << "Mean               : " << stats.mean << "\n";
    cout << "Variance           : " << stats.variance << "\n";
    cout << "Standard Deviation : " << stats.stddev << "\n";
    cout << "Minimum            : " << stats.minimum << "\n";
    cout << "Maximum            : " << stats.maximum << "\n";

    cout << "\n[2] Histogram\n";
    for (size_t i = 0; i < histogram.size(); i++) {
        int startRange = static_cast<int>(i * 1000);
        int endRange = static_cast<int>(((i + 1) * 1000) - 1);
        cout << setw(5) << startRange << " - " << setw(5) << endRange
             << " : " << histogram[i] << "\n";
    }

    cout << "\n[3] Sorting\n";
    cout << "Smallest value     : " << sortedData.front() << "\n";
    cout << "Largest value      : " << sortedData.back() << "\n";
    cout << "Median value       : " << sortedData[sortedData.size() / 2] << "\n";

    cout << "\n[4] Pearson Correlation\n";
    cout << "Correlation value  : " << correlation << "\n";

    cout << "\n[5] Moving Average\n";
    if (!movingAverage.empty()) {
        cout << "First moving avg   : " << movingAverage.front() << "\n";
        cout << "Last moving avg    : " << movingAverage.back() << "\n";
    }

    cout << "\n[6] Outlier Detection\n";
    cout << "Outlier count      : " << outlierCount << "\n";

    cout << "\n========================================\n";
    cout << " Timing Result\n";
    cout << "========================================\n";
    cout << "Data Generation       : " << timing.generationTime << " ms\n";
    cout << "Basic Statistics      : " << timing.statisticsTime << " ms\n";
    cout << "Histogram             : " << timing.histogramTime << " ms\n";
    cout << "Sorting               : " << timing.sortingTime << " ms\n";
    cout << "Pearson Correlation   : " << timing.correlationTime << " ms\n";
    cout << "Moving Average        : " << timing.movingAverageTime << " ms\n";
    cout << "Outlier Detection     : " << timing.outlierTime << " ms\n";
    cout << "Total Runtime         : " << timing.totalTime << " ms\n";
    cout << "========================================\n";
}

int main(int argc, char* argv[]) {
    size_t datasetSize = 1000000;

    if (argc > 1) {
        datasetSize = stoull(argv[1]);
    }

    const int histogramBins = 10;
    const int movingAverageWindow = 5;

    TaskTiming timing{};

    auto totalStart = high_resolution_clock::now();

    auto start = high_resolution_clock::now();
    vector<double> data = generateData(datasetSize);
    vector<double> secondColumn = generateSecondColumn(data);
    auto end = high_resolution_clock::now();
    timing.generationTime = elapsedMs(start, end);

    start = high_resolution_clock::now();
    Statistics stats = calculateStatistics(data);
    end = high_resolution_clock::now();
    timing.statisticsTime = elapsedMs(start, end);

    start = high_resolution_clock::now();
    vector<size_t> histogram = generateHistogram(data, histogramBins);
    end = high_resolution_clock::now();
    timing.histogramTime = elapsedMs(start, end);

    start = high_resolution_clock::now();
    vector<double> sortedData = sortData(data);
    end = high_resolution_clock::now();
    timing.sortingTime = elapsedMs(start, end);

    start = high_resolution_clock::now();
    double correlation = calculatePearsonCorrelation(data, secondColumn);
    end = high_resolution_clock::now();
    timing.correlationTime = elapsedMs(start, end);

    start = high_resolution_clock::now();
    vector<double> movingAverage = calculateMovingAverage(data, movingAverageWindow);
    end = high_resolution_clock::now();
    timing.movingAverageTime = elapsedMs(start, end);

    start = high_resolution_clock::now();
    size_t outlierCount = detectOutliers(data, stats.mean, stats.stddev);
    end = high_resolution_clock::now();
    timing.outlierTime = elapsedMs(start, end);

    auto totalEnd = high_resolution_clock::now();
    timing.totalTime = elapsedMs(totalStart, totalEnd);

    printSummary(datasetSize, stats, histogram, sortedData, correlation,
                 movingAverage, outlierCount, timing);

    saveTimingToCsv("../results/sequential_results.csv", datasetSize, timing);

    cout << "\nTiming saved to ../results/sequential_results.csv\n";

    return 0;
}
