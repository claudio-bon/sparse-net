#include "nn.cuh"
#include "vector_utils.h"
#include "hpc.h"
#include <vector>
#include <string>
#include <iostream>
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
    bool sequential_accumulation = true;
    bool rand = true;
    bool save_output = false;
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
        //Sequential accumulation
        else if (args[i][0] == "-sa" || args[i][0] == "-sacc" || args[i][0] == "--sequential_accumulation") {
            if (args[i][1] == "true") sequential_accumulation = true;
            else if (args[i][1] == "false") sequential_accumulation = false;
            else {
                std::stringstream err;
                err << "Either \"" << args[i][0] << " true\" or \"" << args[i][0] << " false\"";
                throw std::invalid_argument(err.str());
            } 
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


    double start, finish;

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
    

    start = hpc_gettime();
    std::vector<float> output = net.forward(input, sequential_accumulation);
    finish = hpc_gettime();

    //Save forward's output
    if (save_output) {
        std::stringstream filename;
        filename << "output/output_N" << N << "_K" << K << "_R" << R << ".txt";
        write_vector(filename.str(), output);
    }

    std::cout << "Kernel time: "  << net.get_time() << "s" << "\n";
    std::cout << "Nops: "  << net.get_nops() << "\n";
    std::cout << "Total time: "  << finish - start << "s" << "\n";

    return 0;
}
