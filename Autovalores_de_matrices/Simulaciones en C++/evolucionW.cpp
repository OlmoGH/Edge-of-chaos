# include <iostream>
# include <Eigen/Dense>
# include <Eigen/Eigenvalues>
# include <random>
# include <fstream>

class AntiHebbianLearning
{
    private:
        Eigen::MatrixXd connections;
        Eigen::MatrixXd Id;
        Eigen::VectorXd neurons;
        double alpha;
        size_t dim;
        double dt;
    public:
        AntiHebbianLearning(double alpha_, double dt_, size_t dim_);
        void EvolutionConnections();
        void EvolutionNeurons();
        void SaveEigenvalues(std::ofstream& file, int iter);
        void SaveStates(std::ofstream& file, int iter);


};

AntiHebbianLearning::AntiHebbianLearning(double alpha_, double dt_, size_t dim_) : 
alpha(alpha_),
dt(dt_),
dim(dim_)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    auto normal = [&]() { return dist(gen); };

    connections = Eigen::MatrixXd::NullaryExpr(dim_, dim_, normal);
    Id = Eigen::MatrixXd::Identity(dim, dim);
    neurons = Eigen::VectorXd::NullaryExpr(dim_, normal);
}

void AntiHebbianLearning::EvolutionConnections()
{ 
    auto dW_dt = alpha * (Id - neurons * neurons.transpose());
    connections += dW_dt * dt;
}

void AntiHebbianLearning::EvolutionNeurons()
{ 
    auto dx_dt = connections * neurons;
    neurons += dx_dt * dt;
}

void AntiHebbianLearning::SaveEigenvalues(std::ofstream& file, int iter)
{
    Eigen::EigenSolver<Eigen::MatrixXd> solver(connections);
    Eigen::VectorXcd vals = solver.eigenvalues();
    if (iter == 0)
    {
        file << "alpha,dt,dim" << std::endl;
        file << alpha << "," << dt << "," << dim << std::endl;
        
        file << "ev" << 0 << "_re,ev" << 0 << "_im";
        for (size_t k = 1; k < dim; k++) {
            file << ",ev" << k << "_re,ev" << k << "_im";
        }
        file << '\n';
    }

    file << vals[0].real() << ',' << vals[0].imag();
    for (int i = 1; i < vals.size(); ++i) {
        file << ',' << vals[i].real() << ',' << vals[i].imag();
    }
    file << std::endl;
}

void AntiHebbianLearning::SaveStates(std::ofstream& file, int iter)
{

    if (iter == 0)
    {        
        file << "x" << 0;
        for (size_t k = 1; k < dim; k++) {
            file << ",x" << k;
        }
        file << std::endl;
    }

    file << neurons[0];
    for (size_t i = 1; i < dim; ++i) {
        file << ',' << neurons[i];
    }
    file << std::endl;
}

int main() {    
    // Condiciones iniciales
    // - W(0) = matriz gaussiana normalizada
    // - x(0) = ector gaussiano normalizado

    std::ofstream connectFile("Eigenvalues.txt"), neurFile("States.txt");
    if (!connectFile) {std::cerr << "Error loading the connections file" << std::endl; return 1;}
    if (!neurFile) {std::cerr << "Error loading the neurons file" << std::endl; return 1;}

    double alpha = 0.01;
    double dt = 0.01;
    size_t dim = 20;
    int iterations = 100000;

    AntiHebbianLearning state(alpha, dt, dim);

    for (int i = 0; i < iterations; i++)
    {
        state.SaveStates(neurFile, i);
        state.SaveEigenvalues(connectFile, i);
        state.EvolutionNeurons();
        state.EvolutionConnections();
    }

    connectFile.close();
    neurFile.close();
    return 0;
}