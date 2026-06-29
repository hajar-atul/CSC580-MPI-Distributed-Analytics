#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

using namespace std;

double getTimeMs(chrono::high_resolution_clock::time_point start,
                 chrono::high_resolution_clock::time_point end) {
    return chrono::duration<double, milli>(end - start).count();
}

int main(int argc, char* argv[]) {
    long long N = 1000000;
    if (argc > 1) N = atoll(argv[1]);

    cout << "Sequential Analytics Program\n";
    cout << "Dataset size: " << N << "\n";

    vector<double> data(N);
    vector<double> data2(N);

    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0, 10000.0);

    auto totalStart = chrono::high_resolution_clock::now();

    for (long long i = 0; i < N; i++) {
        data[i] = dist(rng);
        data2[i] = data[i] * 0.8 + dist(rng) * 0.2;
    }

    ofstream log("../results/sequential_results.csv", ios::app);
    log << "Task,DatasetSize,Time_ms\n";

    // 1. Basic statistics
    auto start = chrono::high_resolution_clock::now();

    double sum = accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / N;

    double variance = 0.0;
    for (double x : data) variance += (x - mean) * (x - mean);
    variance /= N;

    double stddev = sqrt(variance);
    double minVal = *min_element(data.begin(), data.end());
    double maxVal = *max_element(data.begin(), data.end());

    auto end = chrono::high_resolution_clock::now();
    double statsTime = getTimeMs(start, end);
    log << "Basic Statistics," << N << "," << statsTime << "\n";

    // 2. Histogram
    start = chrono::high_resolution_clock::now();

    int bins = 10;
    vector<int> histogram(bins, 0);

    for (double x : data) {
        int bin = min((int)(x / 1000.0), bins - 1);
        histogram[bin]++;
    }

    end = chrono::high_resolution_clock::now();
    double histTime = getTimeMs(start, end);
    log << "Histogram," << N << "," << histTime << "\n";

    // 3. Sorting
    start = chrono::high_resolution_clock::now();

    vector<double> sortedData = data;
    sort(sortedData.begin(), sortedData.end());

    end = chrono::high_resolution_clock::now();
    double sortTime = getTimeMs(start, end);
    log << "Sorting," << N << "," << sortTime << "\n";

    // 4. Pearson correlation
    start = chrono::high_resolution_clock::now();

    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

    for (long long i = 0; i < N; i++) {
        sumX += data[i];
        sumY += data2[i];
        sumXY += data[i] * data2[i];
        sumX2 += data[i] * data[i];
        sumY2 += data2[i] * data2[i];
    }

    double numerator = (N * sumXY) - (sumX * sumY);
    double denominator = sqrt(((N * sumX2) - (sumX * sumX)) *
                              ((N * sumY2) - (sumY * sumY)));

    double correlation = numerator / denominator;

    end = chrono::high_resolution_clock::now();
    double corrTime = getTimeMs(start, end);
    log << "Pearson Correlation," << N << "," << corrTime << "\n";

    // 5. Moving average
    start = chrono::high_resolution_clock::now();

    int window = 5;
    vector<double> movingAverage;

    double windowSum = 0.0;
    for (long long i = 0; i < N; i++) {
        windowSum += data[i];

        if (i >= window) windowSum -= data[i - window];

        if (i >= window - 1) {
            movingAverage.push_back(windowSum / window);
        }
    }

    end = chrono::high_resolution_clock::now();
    double movingTime = getTimeMs(start, end);
    log << "Moving Average," << N << "," << movingTime << "\n";

    // 6. Outlier detection
    start = chrono::high_resolution_clock::now();

    int outlierCount = 0;
    double lowerBound = mean - 3 * stddev;
    double upperBound = mean + 3 * stddev;

    for (double x : data) {
        if (x < lowerBound || x > upperBound) outlierCount++;
    }

    end = chrono::high_resolution_clock::now();
    double outlierTime = getTimeMs(start, end);
    log << "Outlier Detection," << N << "," << outlierTime << "\n";

    auto totalEnd = chrono::high_resolution_clock::now();
    double totalTime = getTimeMs(totalStart, totalEnd);
    log << "Total," << N << "," << totalTime << "\n";
    log.close();

    cout << "\n=== Basic Statistics ===\n";
    cout << "Mean: " << mean << "\n";
    cout << "Variance: " << variance << "\n";
    cout << "Standard Deviation: " << stddev << "\n";
    cout << "Minimum: " << minVal << "\n";
    cout << "Maximum: " << maxVal << "\n";

    cout << "\n=== Histogram ===\n";
    for (int i = 0; i < bins; i++) {
        cout << i * 1000 << "-" << ((i + 1) * 1000 - 1)
             << ": " << histogram[i] << "\n";
    }

    cout << "\n=== Sorting ===\n";
    cout << "Smallest value: " << sortedData.front() << "\n";
    cout << "Largest value: " << sortedData.back() << "\n";

    cout << "\n=== Pearson Correlation ===\n";
    cout << "Correlation: " << correlation << "\n";

    cout << "\n=== Moving Average ===\n";
    cout << "First moving average: " << movingAverage.front() << "\n";
    cout << "Last moving average: " << movingAverage.back() << "\n";

    cout << "\n=== Outlier Detection ===\n";
    cout << "Outlier count: " << outlierCount << "\n";

    cout << "\n=== Timing ===\n";
    cout << "Basic Statistics Time: " << statsTime << " ms\n";
    cout << "Histogram Time: " << histTime << " ms\n";
    cout << "Sorting Time: " << sortTime << " ms\n";
    cout << "Pearson Correlation Time: " << corrTime << " ms\n";
    cout << "Moving Average Time: " << movingTime << " ms\n";
    cout << "Outlier Detection Time: " << outlierTime << " ms\n";
    cout << "Total Time: " << totalTime << " ms\n";

    cout << "\nResults saved to ../results/sequential_results.csv\n";

    return 0;
}