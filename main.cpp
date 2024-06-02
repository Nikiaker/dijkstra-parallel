#include <iostream>
#include <ctime>

#include "graph.cpp"

int main()
{
    std::srand(std::time(nullptr));
    int v;
    int a, b;
    std::cin >> v >> a >> b;
    Graph graph(v);
    //graph.show_adj();
    std::vector<int> road = graph.find_shortest_path_openmp(a, b);
    graph.convert_to_adj(road);
    graph.show_adj();
}