// anti_hebbian.h
# ifndef ANTI_HEBBIAN_H
# define ANTI_HEBBIAN_H

# include <Eigen/Dense>
# include <fstream>

enum MatrixType {Normal=1, Symmetric=2, Antisymetric=3, Diagonal=4};
enum EvolAlgorythm {Euler=1, RK4=2};

struct ConfigSimulacion {
    size_t dim = 0;
    double alpha = 0;
    double dt = 0;
    int iterations = 0;
    int seed_connections = 0;
    int seed_neurons = 0;
    std::ofstream* connectFile = nullptr;
    std::ofstream* neurFile = nullptr;
    std::ofstream* deltaFile = nullptr;
    MatrixType initial_connections = Normal;
    EvolAlgorythm evolution_algorythm = Euler;
    double mean_connections = 0;
    double stdev_connections = 1;
};

class AntiHebbianLearning
{
    private:
        Eigen::MatrixXd connections;
        Eigen::MatrixXd Id;
        Eigen::VectorXd neurons;
        double alpha;
        size_t dim;
        double dt;
        size_t iterations;
    public:
        AntiHebbianLearning(ConfigSimulacion mi_config);
        Eigen::MatrixXd dW_dt(Eigen::VectorXd vector_neurons);
        Eigen::VectorXd dx_dt(Eigen::MatrixXd matrix_connections, Eigen::VectorXd vector_neurons);
        void IntegrateStep(ConfigSimulacion mi_config);
        void ExportEigenvalues(ConfigSimulacion mi_config);
        void ExportState(ConfigSimulacion mi_config);
        Eigen::VectorXd get_neurons();
        Eigen::MatrixXd get_connections();
        void set_neurons(Eigen::VectorXd newNeurons);
        void set_connections(Eigen::MatrixXd newConections);

};

bool InitializeFiles(ConfigSimulacion mi_config);
void CloseFiles(ConfigSimulacion mi_config);
void ExportData(ConfigSimulacion mi_config, AntiHebbianLearning& state1, AntiHebbianLearning* state2=nullptr);
void ExportDeltas(ConfigSimulacion mi_config, AntiHebbianLearning& state1, AntiHebbianLearning& state2);

#endif