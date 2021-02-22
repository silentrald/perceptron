#include "../include/perceptron.h"

Perceptron::Perceptron(int inputs, double learning_rate) {
    weights.resize(inputs);
    this->learning_rate = learning_rate;
}

void Perceptron::zero_weights() {
    for (int i = weights.size() - 1; i > -1; i--) {
        weights[i] = 0;
    }
}

void Perceptron::randomize_weights() {
    for (int i = weights.size() - 1; i > -1; i--) {
        weights[i] = drand48();
    }
}

int Perceptron::predict(std::vector<double> inputs) {
    if (inputs.size() != weights.size()) return 0;

    double output = 0.0;

    for (int i = weights.size() - 1; i > -1; i--) {
        output += weights[i] * inputs[i];
    }

    return output < 0.0 ? -1 : 1;
}

void Perceptron::train(std::vector<double> inputs, int label) {
    if (inputs.size() != weights.size()) return;

    double output = 0.0;

    for (int i = weights.size() - 1; i > -1; i--) {
        output += weights[i] * inputs[i];
    }

    // y = label
    // y hat = (output < 0.0 ? -1 : 1)
    int error = label - (output < 0.0 ? -1 : 1);

    if (!error) return;

    // adjust if the error is not zero
    for (int i = weights.size() - 1; i > -1; i--) {
        weights[i] = weights[i] + learning_rate * error * inputs[i];
    }
}

void Perceptron::print() {
    printf("Perceptron\n");
    printf("Learning Rate: %lf\n", learning_rate);
    printf("Weights:\n");
    for (int i = 0; i < weights.size(); i++) {
        printf("%d: %lf\n", i, weights[i]);
    }
}

void Perceptron::to_file(std::string filename) {
    FILE* fp = fopen(filename.c_str(), "w");

    fprintf(fp, "Perceptron\n");
    fprintf(fp, "Learning Rate: %lf\n", learning_rate);
    fprintf(fp, "Weights:\n");
    for (int i = 0; i < weights.size(); i++) {
        fprintf(fp, "%d: %lf\n", i, weights[i]);
    }
}
