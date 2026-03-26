# include "lib/anti_hebbian.h"
# include <iostream>


int main() {
    // Condiciones iniciales
    // - W(0) = matriz gaussiana normalizada
    // - x(0) = vector gaussiano normalizado

    // Archivos de salida de los datos
    std::ofstream archivoConexiones;
    std::ofstream archivoNeuronas;
    std::ofstream archivoDeltas;

    // Especificamos la cinfiguración y parámetros del sistema
    ConfigSimulacion mi_config;
    mi_config.alpha = 0.001; // Ratio de aprendizaje
    mi_config.dt = 0.01; // Intervalo de tiempo entre pasos
    mi_config.dim = 300; // Tamaño del vector de neuronas
    mi_config.iterations = 1000000; // Iteraciones de la evolución
    mi_config.seed_connections = 1213; // Semilla para crear la matriz de conexiones
    mi_config.seed_neurons = 21; // Semilla para crear la matriz de neuronas
    mi_config.mean_connections = 0.0; // Valor medio de las entradas de la matriz de conexiones
    mi_config.stdev_connections = 1 / std::sqrt(mi_config.dim); // Desviación estándar de las entradas de la matriz de conexiones
    mi_config.initial_connections = Normal; // Configuración inicial de la matriz de conexiones
    mi_config.evolution_algorythm = RK4; // Algoritmo de integración 

    // Especificamos los archivos de salida si queremos que haya output
    // mi_config.connectFile = &archivoConexiones;
    mi_config.neurFile = &archivoNeuronas;
    // mi_config.deltaFile = &archivoDeltas;

    // Intentamos inicializar los archivos
    if (!InitializeFiles(mi_config)) {
        return -1; // Salir si hubo error
    }

    // Creamos las dos redes neuronales
    AntiHebbianLearning state1(mi_config);
    // AntiHebbianLearning state2(mi_config);

    // Le añadimos una pequeña diferencia a la segunda red de neuronas
    // Eigen::VectorXd neurons2 = state2.get_neurons();
    // Eigen::VectorXd delta = Eigen::VectorXd::Ones(mi_config.dim) * 0.001;
    // state2.set_neurons(neurons2 + delta);

    // Evolucionamos ambos sistemas de forma paralela
    for (int i = 0; i < mi_config.iterations; i++)
    {
        // Guardamos todos los datos en cada paso
        if (i % 100 == 0) ExportData(mi_config, state1);

        // Evolucionamos el sistema y la matriz de conexiones
        state1.IntegrateStep(mi_config);
        // state2.IntegrateStep(mi_config);
        if (i % 10000 == 0) 
        {
            std::cout << "Iteracion " << i << std::endl;
        }
    }

    CloseFiles(mi_config);
    return 0;
}