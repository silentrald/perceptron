#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <stdio.h>
#include <time.h>
#include "./include/perceptron.h"

struct Data {
    int label;
    double diameter;
    double red;
};

int main() {
    srand(time(NULL));
    srand48(time(NULL));

    Perceptron perceptron(2);
    // perceptron.randomize_weights();
    perceptron.zero_weights();

    // load dataset
    std::string line;
    std::ifstream fs("dataset/clean.csv");

    if (!fs.is_open()) {
        printf("File could not be opened\n");
        return 1;
    }

    // skip the first line
    getline(fs, line);

    std::vector<Data> dataset;
    std::string token;
    int label;
    double diameter, red;

    while (getline(fs, line)) {
        std::stringstream ss(line);

        std::getline(ss, token, ',');
        label = token == "orange" ? 1 : -1;

        std::getline(ss, token, ',');
        diameter = atof(token.c_str());

        std::getline(ss, token, ',');
        red = atof(token.c_str());

        Data data;
        data.label = label;
        data.diameter = diameter;
        data.red = red;

        dataset.push_back(data);
    }
    fs.close();

    random_shuffle(dataset.begin(), dataset.end());
    random_shuffle(dataset.begin(), dataset.end());
    random_shuffle(dataset.begin(), dataset.end());

    // Split to train and test
    // 80% train 20% test
    int size = dataset.size();
    int train_size = size * 0.80;
    int test_size = size - train_size;

    // Train
    Data data;
    for (int i = 0; i < train_size; i++) {
        data = dataset[i];

        perceptron.train({
            data.diameter,
            data.red
        }, data.label);
    }

    std::vector<std::vector<int>> confusion_matrix = {
        { 0, 0, 0 },
        { 0, 0, 0 },
        { 0, 0, 0 }
    };

    int guess;
    for (int i = 0; i < test_size; i++) {
        data = dataset[i + train_size];

        guess = perceptron.predict({
            data.diameter,
            data.red
        });
        
        int r = guess > 0 ? 0 : 1;
        int c = data.label > 0 ? 0 : 1;
        confusion_matrix[r][c]++;
    }

    int sum;
    for (int r = 0; r < 2; r++) {
        sum = 0;
        for (int c = 0; c < 2; c++) {
            sum += confusion_matrix[r][c];
        }
        confusion_matrix[r][2] = sum;
    }

    for (int c = 0; c < 2; c++) {
        sum = 0;
        for (int r = 0; r < 2; r++) {
            sum += confusion_matrix[r][c];
        }
        confusion_matrix[2][c] = sum;
    }

    confusion_matrix[2][2] = test_size;

    printf("Confusion Matrix\n");
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            printf("%5d", confusion_matrix[r][c]);
        }
        printf("\n");
    }

    perceptron.to_file();
}