#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <fstream>
#include <list>

#include "graph.cpp"

void print_vector(std::vector<int> vec)
{
    for (int x : vec)
    {
        std::cout << x << " ";
    }
}

template <class T>
long long avg(T tab, int n)
{
    long long sum = 0;
    for (int i = 0; i < n; i++)
        sum += tab[i];
    return sum / n;
}

template <class T>
void convert_json(T tab, int n, std::string name)
{
    std::ofstream file(name);
    file << "{" << std::endl;
    for (int i = 0; i < n; i++)
    {
        file << "\t\"" << i + 2 << "\": " << tab[i] << "," << std::endl;
    }
    file << "}";
    file.close();
}

std::vector<Graph> generate_graphs(int n, int graph_size)
{
    std::vector<Graph> graphs;
    for (int i = 0; i < n; i++)
    {
        Graph g(graph_size);
        graphs.push_back(g);
    }
    return graphs;
}

int main()
{
    std::srand(std::time(nullptr));

    const int tests = 2000;
    const int graph_size = 100;
    const int acc = 1;

    std::vector<Graph> graphs = generate_graphs(tests, graph_size);

    std::vector<long long> fin_res(tests);

    // iteration
    for (int i = 0; i < graphs.size(); i++)
    {
        std::vector<long long> res;
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < i + 1; j++)
        {
            graphs[j].find_shortest_path(0, 1);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        res.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
        fin_res[i] = avg<std::vector<long long>>(res, res.size());
    }
    convert_json<std::vector<long long>>(fin_res, fin_res.size(), "iteration.json");

    // openmp
    for (int i = 0; i < graphs.size(); i++)
    {
        std::vector<long long> res;
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for shared(graphs)
        for (int j = 0; j < i; j++)
        {
            graphs[j].find_shortest_path(0, 1);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        res.push_back(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
        fin_res[i] = avg<std::vector<long long>>(res, res.size());
    }
    convert_json<std::vector<long long>>(fin_res, fin_res.size(), "openmp.json");
}