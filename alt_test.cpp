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

std::vector<Graph> generate_graphs(int n)
{
    std::vector<Graph> graphs;
    for (int i = 2; i < n + 2; i++)
    {
        Graph g(i);
        graphs.push_back(g);
    }
    return graphs;
}

int main()
{
    std::srand(std::time(nullptr));

    const int tests = 2000;
    const int acc = 1;

    std::vector<Graph> graphs = generate_graphs(tests);

    std::vector<long long> fin_res(tests);

    // iteration
    for (int i = 0; i < graphs.size(); i++)
    {
        std::vector<long long> res(acc);
        for (int j = 0; j < acc; j++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            graphs[i].find_shortest_path(0, 1);
            auto stop = std::chrono::high_resolution_clock::now();
            res[j] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        }
        fin_res[i] = avg<std::vector<long long>>(res, res.size());
    }
    convert_json<std::vector<long long>>(fin_res, fin_res.size(), "iteration.json");

    // openmp
    for (int i = 0; i < graphs.size(); i++)
    {
        std::vector<long long> res(acc);
        for (int j = 0; j < acc; j++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            graphs[i].find_shortest_path_openmp(0, 1);
            auto stop = std::chrono::high_resolution_clock::now();
            res[j] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        }
        fin_res[i] = avg<std::vector<long long>>(res, res.size());
    }
    convert_json<std::vector<long long>>(fin_res, fin_res.size(), "openmp.json");
}