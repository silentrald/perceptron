#include <vector>
#include <random>
#include <stdio.h>

class Perceptron {
private:
    std::vector<double> weights;
    double learning_rate;

public:
    // CONSTRUCTOR
    Perceptron(int inputs, double learning_rate = 0.001);

    // METHODS
    void zero_weights();
    void randomize_weights();

    int predict(std::vector<double> inputs);
    void train(std::vector<double> inputs, int label);

    void print();
    void to_file(std::string filename = "perceptron.txt");
};