#include "nn.h"
#include "vector_utils.h"
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <stdexcept>
#include <sstream>


int main(int argc, char* argv[]) {

    // argc-1 because the name of the program is included as well
    if ((argc-1) % 2 != 0) exit(1);

    //Prepare the arguments
    std::vector<std::vector<std::string> > args((argc-1)/2);
    for (std::size_t i = 1, j = 0; i < argc && j < args.size(); j++) {
        args[j] = std::vector<std::string>(2);
        for (std::size_t k = 0; i < argc && k < args[j].size(); i++, k++) {
            args[j][k] = std::string(argv[i]);
        }
    }

    //Default arguments
    std::size_t N, K;
    bool ISN=false, ISK=false;
    parallelism_t parallelism = OUTER;
    bool rand = true, save_output = false;
    std::string net_filename, input_filename;
    bool net_file = false, input_file=false;

    //Read the arguments
    for (std::size_t i = 0; i < args.size(); i++) {
        //Input size
        if (args[i][0] == "-n" || args[i][0] == "-N" || args[i][0] == "--input_size" || args[i][0] == "--first_layer_size") {
            N = std::stoi(args[i][1]);
            ISN = true;
        }
        //Number of Layers
        else if (args[i][0] == "-k" || args[i][0] == "-K" || args[i][0] == "--num_layers") {
            K = std::stoi(args[i][1]);
            ISK = true;
        }
        //Parallelism type
        else if (args[i][0] == "-pt" || args[i][0] == "-PT" || args[i][0] == "--parallelism_type") {
            if (args[i][1] == "FULL" || args[i][1] == "full" || args[i][1] == "Full")
                parallelism = FULL;
            else if (args[i][1] == "OUTER" || args[i][1] == "outer" || args[i][1] == "Outer")
                parallelism = OUTER;
            else if (args[i][1] == "INNER" || args[i][1] == "inner" || args[i][1] == "Inner")
                parallelism = INNER;
            else if (args[i][1] == "SEQUENTIAL" || args[i][1] == "sequential" || args[i][1] == "Sequential" ||
                     args[i][1] == "SEQ" || args[i][1] == "seq" || args[i][1] == "Seq")
                parallelism = SEQUENTIAL;
            else
                throw std::invalid_argument("Uknown parallelism type");
        }
        //Random computation
        else if (args[i][0] == "-rd" || args[i][0] == "--random") {
            if (args[i][1] == "true") rand = true;
            else if (args[i][1] == "false") rand = false;
            else {
                std::stringstream err;
                err << "Either \"" << args[i][0] << " true\" or \"" << args[i][0] << " false\"";
                throw std::invalid_argument(err.str());
            }
        }
        //Save output
        else if (args[i][0] == "-so" || args[i][0] == "--save_output") {
            if (args[i][1] == "true") save_output = true;
            else if (args[i][1] == "false") save_output = false;
            else {
                std::stringstream err;
                err << "Either \"" << args[i][0] << " true\" or \"" << args[i][0] << " false\"";
                throw std::invalid_argument(err.str());
            }
        }
        //Load Neural Network from file
        else if (args[i][0] == "-nf" || args[i][0] == "--net_file") {
            net_filename = args[i][1];
            net_file = true;
        }
        //Load input from file
        else if (args[i][0] == "-if" || args[i][0] == "--input_file") {
            input_filename = args[i][1];
            input_file = true;
        }
        //Uknown argument
        else throw std::invalid_argument("Uknown argument passed to the program");
    }

    //Check mandatory arguments
    if (!net_file || !input_file) {
        if (net_file && !ISN)
            throw std::invalid_argument("The size of the input (-n) must be specified");
        else if (!(net_file && ISN) && (!ISN || !ISK))
            throw std::invalid_argument("Both the size of the input (-n) and the number of Layers (-k) must be specified");
    }

    if (ISK && K <= 1) throw std::invalid_argument("Layers must be at least 2");


    std::vector<float> input;
    if (!input_file)
        input = create_vector<float>(N, rand);
    else
        input = read_vector<float>(input_filename);
    
    NN<float> net;
    if (!net_file)
        net = NN<float>(N, K, rand);
    else
        net = NN<float>(net_filename);
    

    double time = omp_get_wtime();
    std::vector<float> output = net.forward(input, parallelism);
    time = omp_get_wtime() - time;

    //Save forward's output
    if (save_output) {
        std::stringstream filename;
        filename << "output/output_N" << N << "_K" << K << "_R" << R << ".txt";
        write_vector(filename.str(), output);
    }

    std::cout << "Execution time: " << time << "s" << "\n";

    return 0;
}
