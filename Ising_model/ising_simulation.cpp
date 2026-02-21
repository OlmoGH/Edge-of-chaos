# include <iostream>
# include <Eigen/Dense>
# include <Eigen/Eigenvalues>
# include <random>
# include <fstream>
# include <vector>
# include <chrono>

class IsingSimulation 
{
    public:
        double T;
        int dim;
        int iterations;
        Eigen::MatrixXi lattice;
        Eigen::MatrixXd TimeSeries;
        std::ofstream *file;
        std::mt19937 rng;
        std::uniform_real_distribution<> dist_prob; // Para la probabilidad
        std::uniform_int_distribution<> dist_dim; // Para coordenadas x, y

        IsingSimulation(double T_, int dim_, std::ofstream *file_);
        void CrossMontecarloStep();
        void FullMontecarloStep();
        void SaveState();
        Eigen::VectorXd CalculateCovarianceEigenvalues();
};

IsingSimulation::IsingSimulation(double T_, int dim_, std::ofstream *file_): 
T(T_), 
dim(dim_), 
file(file_),
rng(std::chrono::system_clock::now().time_since_epoch().count()),
dist_prob(0.0, 1.0),
dist_dim(0, dim - 1)
{
    // Inicializa el lattice a -1 y 1
    lattice = ((Eigen::MatrixXi::Random(dim_, dim_).array() < 0).cast<int>() * 2 - 1).matrix();
    TimeSeries = Eigen::MatrixXd(50000, dim * dim);
}
void IsingSimulation::CrossMontecarloStep()
{
    // Calcula la probabilidad de transición de una casilla random
    int x = dist_dim(rng);
    int y = dist_dim(rng);

    double E = -2 * lattice(x, y) * (
        lattice(x, (y - 1 + dim) % dim) +
        lattice(x, (y + 1 + dim) % dim) +
        lattice((x - 1 + dim) % dim, y) +
        lattice((x + 1 + dim) % dim, y)
        );
    
    double prob = exp(E / T);
    
    if (dist_prob(rng) < prob) lattice(x, y) = -lattice(x, y);
}
void IsingSimulation::FullMontecarloStep()
{
    // Calcula la probabilidad de transición de una casilla random
    int x = dist_dim(rng);
    int y = dist_dim(rng);

    double contorno = 0;
    for (int i = -1; i < 2; i++)
    {
        for (int j = -1; j < 2; j++)
        {
            if (i * j == i + j) contorno += lattice((x+i+dim)%dim, (y+j+dim)%dim);
        }
    }

    double dE = -2 * lattice(x, y) * contorno;
    
    double prob = exp(dE / T);
    
    if (dist_prob(rng) < prob) lattice(x, y) = -lattice(x, y);
}
void IsingSimulation::SaveState()
{
    *file << lattice << std::endl;
}
Eigen::VectorXd IsingSimulation::CalculateCovarianceEigenvalues()
{
    if (TimeSeries.rows() < 2)
    {
        std::cout << "No hay suficientes datos para calcular la covarianza" << std::endl;
        return Eigen::MatrixXd::Zero(dim * dim, dim * dim);
    }
    
    std::cout << "He entrado en la funcion" << std::endl;

    // Centramos la serie temporal en torno a la media
    Eigen::MatrixXd X = TimeSeries.rowwise() - TimeSeries.colwise().mean();
    
    std::cout << "He centrado la serie en torno a su media" << std::endl;

    Eigen::MatrixXd Cov = (X.transpose() * X) / (TimeSeries.rows() - 1);

    std::cout << "He calculado la matriz de covarianza" << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solve(Cov);

    if (solve.info() != Eigen::Success) {
        // Manejo de error si la descomposición falló
        std::cerr << "ERROR: La descomposición de autovalores falló." << std::endl;
        return Eigen::VectorXd::Zero(dim * dim);
    }

    return solve.eigenvalues();
}
int main()
{
    std::ofstream output("States.txt");
    if(!output) {std::cerr << "Error al abrir el archivo" << std::endl; return 1;}
    std::ofstream covFile("CovarianceEigenvalues.txt");
    if(!covFile) {std::cerr << "Error al abrir el archivo" << std::endl; return 1;}

    double T = 2.0 / log(1+sqrt(2)) + 0.1;
    int dim = 30;
    int iteraciones = 1000000;
    int skip = 100;
    int termalizacion = 100000;
    IsingSimulation simulacion(T, dim, &output);
    
    std::cout << std::endl << std::endl;

    for (int i = 0; i<iteraciones; i++)
    {
        // Guardamos el estado
        if ((i % skip == 0) && (i > termalizacion)) 
        {
            simulacion.TimeSeries.row(i / skip) = simulacion.lattice.reshaped(1, dim * dim).cast<double>();
            std::cout << i* 100 / iteraciones  << "%" << '\r';
            simulacion.SaveState();
        }
        // Aplicamos Montecarlo
        simulacion.CrossMontecarloStep();
    }

    std::cout << "Voy a exportar la matriz" << std::endl;
    covFile << simulacion.CalculateCovarianceEigenvalues();;
    std::cout << "Matriz exportada" << std::endl;

    return 0;
}