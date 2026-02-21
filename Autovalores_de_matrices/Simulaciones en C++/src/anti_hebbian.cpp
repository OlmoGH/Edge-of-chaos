// anti_hebbian.cpp
# include "../lib/anti_hebbian.h"
# include <iostream>
# include <Eigen/Dense>
# include <Eigen/Eigenvalues>
# include <random>
# include <fstream>
# include <iomanip>
# include <sstream>
# include <string>

AntiHebbianLearning::AntiHebbianLearning(ConfigSimulacion mi_config) : 
alpha(mi_config.alpha),
dt(mi_config.dt),
dim(mi_config.dim),
iterations(mi_config.iterations)
{
    // Creación del vector de neuronas mediante una distribución gaussiana N(0, 1)
    std::mt19937 gen_neurons(mi_config.seed_neurons);
    std::normal_distribution<double> dist_neurons(0.0, 1.0);
    auto lambda_neurons = [&]() { return dist_neurons(gen_neurons); };
    std::mt19937 gen_connections(mi_config.seed_connections);
    std::normal_distribution<double> dist_connections(mi_config.mean_connections, mi_config.stdev_connections);
    auto lambda_connections = [&]() { return dist_connections(gen_connections); };
    Eigen::MatrixXd normal_matrix = Eigen::MatrixXd::NullaryExpr(mi_config.dim, mi_config.dim, lambda_connections);

    // Elegimos la configuración inicial de la matriz de conexiones
    switch (mi_config.initial_connections)
    {
    case Normal:
        // Matriz con cada entrada dada por una distribución normal
        connections = normal_matrix;
        break;
    case Symmetric:
        // Matriz con cada entrada dada por una distribución normal y luego simetrizada
        connections = 0.5 * (normal_matrix + normal_matrix.transpose());
        break;
    case Antisymetric:
        // Matriz con cada entrada dada por una distribución normal y luego antisimetrizada
        connections = 0.5 * (normal_matrix - normal_matrix.transpose());
        break;
    case Diagonal:
        // Matriz diagonal con valor mi_config.mean_connections
        connections = Eigen::MatrixXd::Identity(mi_config.dim, mi_config.dim) * mi_config.mean_connections;
        break;
    default:
        // La configuración por defecto es la matriz normal
        connections = normal_matrix * 0.0;
    }




    // Inicialización de los elementos del sistema
    Id = Eigen::MatrixXd::Identity(mi_config.dim, mi_config.dim);

    //Inicializamos las neuronas de forma aleatoria
    neurons = Eigen::VectorXd::NullaryExpr(mi_config.dim, lambda_neurons);

}

Eigen::MatrixXd AntiHebbianLearning::dW_dt(Eigen::VectorXd vector_neurons)
{
    return alpha * (Id - vector_neurons * vector_neurons.transpose());

}
Eigen::VectorXd AntiHebbianLearning::dx_dt(Eigen::MatrixXd matrix_connections, Eigen::VectorXd vector_neurons)
{
    return matrix_connections * vector_neurons;
}

void AntiHebbianLearning::IntegrateStep(ConfigSimulacion mi_config)
{  
    switch (mi_config.evolution_algorythm)
    {
    case Euler:
        connections += dW_dt(neurons) * dt;
        neurons += dx_dt(connections, neurons) * dt;
        break;
    
    case RK4:
    {
        Eigen::MatrixXd kW1 = dW_dt(neurons);
        Eigen::VectorXd kx1 = dx_dt(connections, neurons);
        Eigen::MatrixXd kW2 = dW_dt(neurons + dt * 0.5 * kx1);
        Eigen::VectorXd kx2 = dx_dt(connections + dt * 0.5 * kW1, neurons + dt * 0.5 * kx1);
        Eigen::MatrixXd kW3 = dW_dt(neurons + dt * 0.5 * kx2);
        Eigen::VectorXd kx3 = dx_dt(connections + dt * 0.5 * kW2, neurons + dt * 0.5 * kx2);
        Eigen::MatrixXd kW4 = dW_dt(neurons + dt * kx3);
        Eigen::VectorXd kx4 = dx_dt(connections + dt * kW3, neurons + dt * kx3);
        connections += dt / 6.0 * (kW1 + 2 * kW2 + 2 * kW3 + kW4);
        neurons += dt / 6.0 * (kx1 + 2 * kx2 + 2 * kx3 + kx4);
        break;
    }
    default:
        break;
    }
}

Eigen::VectorXd AntiHebbianLearning::get_neurons()
{
    return neurons;
}

Eigen::MatrixXd AntiHebbianLearning::get_connections()
{
    return connections;
}

void AntiHebbianLearning::set_neurons(Eigen::VectorXd newNeurons)
{
    if (neurons.size() != newNeurons.size()){
        std::cerr << "Los vectores de neuronas no tienen el mismo tamaño" << std::endl;
    }
    neurons = newNeurons;
}

void AntiHebbianLearning::set_connections(Eigen::MatrixXd newConections)
{
    if (connections.size() != newConections.size()){
        std::cerr << "Las matrices de conexiones no tienen el mismo tamaño" << std::endl;
    }
    connections = newConections;
}

void AntiHebbianLearning::ExportEigenvalues(ConfigSimulacion mi_config)
{
    // 1. Calcular los autovalores de la matriz actual 'connections'
    // EigenSolver devuelve un objeto que contiene autovalores y autovectores
    Eigen::EigenSolver<Eigen::MatrixXd> solver(connections);
    
    // Extraemos solo los autovalores (devuelve un VectorXcd -> Complejos)
    // El formato en memoria de esto ya es: [Re0, Im0, Re1, Im1, ..., ReN, ImN]
    Eigen::VectorXcd autovalores = solver.eigenvalues();

    // 2. Calcular el tamaño en bytes
    // sizeof(std::complex<double>) suele ser 16 bytes (8 double real + 8 double imag)
    std::streamsize tamano_bytes = autovalores.size() * sizeof(std::complex<double>);

    // 3. Escribir al archivo binario
    mi_config.connectFile->write(reinterpret_cast<const char*>(autovalores.data()), tamano_bytes);
}

void AntiHebbianLearning::ExportState(ConfigSimulacion mi_config)
{
    // 1. Calculamos el tamaño del vector en bytes
    // neurons.size() es el número de neuronas (N)
    // sizeof(double) son 8 bytes por número
    std::streamsize tamano_bytes = neurons.size() * sizeof(double);

    // 2. Volcamos la memoria cruda al archivo
    // neurons.data() nos da el puntero al array de doubles
    mi_config.neurFile->write(reinterpret_cast<const char*>(neurons.data()), tamano_bytes);
}

void ExportDeltas(ConfigSimulacion mi_config, AntiHebbianLearning& state1, AntiHebbianLearning& state2)
{
    // 1. Calculamos el tamaño del vector en bytes
    std::streamsize tamano_bytes = state1.get_neurons().size() * sizeof(double);

    // 2. Volcamos la memoria cruda al archivo
    Eigen::VectorXd deltas = state1.get_neurons() - state2.get_neurons();
    mi_config.deltaFile->write(reinterpret_cast<const char*>(deltas.data()), tamano_bytes);
}

void ExportData(ConfigSimulacion mi_config, AntiHebbianLearning& state1, AntiHebbianLearning* state2)
{
    if (mi_config.connectFile != nullptr)
    {
        state1.ExportEigenvalues(mi_config);
    }    
    
    if (mi_config.neurFile != nullptr)
    {
        state1.ExportState(mi_config);
    }

    if (mi_config.deltaFile!= nullptr)
    {
        ExportDeltas(mi_config, state1, *state2);
    }

}

bool InitializeFiles(ConfigSimulacion mi_config)
{
    // 1. Crear el nombre del archivo de forma más limpia (C++ style)
    std::stringstream ss;
    ss << std::fixed << std::setprecision(int(-std::log10(mi_config.alpha))) << mi_config.alpha << "_" 
       << mi_config.dim << "_" 
       << std::fixed << std::setprecision(int(-std::log10(mi_config.dt))) << mi_config.dt;
    
    std::string nameParameters = ss.str();
    std::string nameEigenvalues = "output/Eigenvalues_" + nameParameters + ".bin";
    std::string nameStates = "output/States_" + nameParameters + ".bin";
    std::string nameDelta = "output/Delta_" + nameParameters + ".bin";

    // 2. Abrir los archivos (usando los punteros pasados)
    if (mi_config.connectFile != nullptr){
        mi_config.connectFile->open(nameEigenvalues, std::ios::binary);
        if (!mi_config.connectFile->is_open()) {
            std::cerr << "Error: No se pudo crear el archivo " << nameEigenvalues << std::endl;
            return false;
        }
    }

    if (mi_config.neurFile != nullptr){
        mi_config.neurFile->open(nameStates, std::ios::binary);
        if (!mi_config.neurFile->is_open()) {
            std::cerr << "Error: No se pudo crear el archivo " << nameStates << std::endl;
            return false;
        }
    } 

    if (mi_config.deltaFile != nullptr){
        mi_config.deltaFile->open(nameDelta, std::ios::binary);
        if (!mi_config.deltaFile->is_open()) {
            std::cerr << "Error: No se pudo crear el archivo " << nameDelta << std::endl;
            return false;
        }
    } 


    return true; // Todo salió bien
}

void CloseFiles(ConfigSimulacion mi_config)
{
    // Cerrar los archivos (usando los punteros pasados)
    if (mi_config.connectFile != nullptr){
        mi_config.connectFile->close();
    }

    if (mi_config.neurFile != nullptr){
        mi_config.neurFile->close();
    } 

    if (mi_config.deltaFile != nullptr){
        mi_config.deltaFile->close();
    } 
}