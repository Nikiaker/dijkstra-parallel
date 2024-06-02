#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <climits>
#include <queue>
#include <algorithm>
#include <list>
#include <omp.h>

class Graph
{
    struct Edge
    {
        int a;  // wierzchołek a
        int b;  // wierzchołek b
        int w;  // waga łuku
        char c; // kolor łuku
    };

    int** matrix;           // macierz sąsiedztwa
    std::vector<Edge> adj;  // lista łuków
    int vertices;           // ilość wierzchołków w grafie
    int edges = 0;          // ilość łuków w grafie

    // Creates a matrix of size n*n and returns a pointer
    int** create_matrix(int n)
    {
        int** matrix = new int*[n];
        for (int i = 0; i < n; i++)
            matrix[i] = new int[n];
        return matrix;
    }

    // Fills a matrix with 0s
    void fill_zeros(int** matrix, int n)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i][j] = 0;
    }

    // Generate a random number in selected range
    int random_num(int from, int to)
    {
        int diff = to - from;
        return from + std::rand() % (diff + 1);
    }

    void fill_false(bool* b, int n)
    {
        for (int i = 0; i < n; i++)
            b[i] = false;
    }

    void generate_directed_graph()
    {
        for (int i = 1; i < this->vertices; i++)
            for (int j = 0; j < i; j++)
                if (random_num(0, 1))
                {
                    this->matrix[i][j] = random_num(1, 100);
                    this->edges++;
                }
    }

    void convert_to_undirected()
    {
        for (int i = 1; i < this->vertices; i++)
            for (int j = 0; j < i; j++)
                this->matrix[j][i] = this->matrix[i][j];
    }

    void fix_matrix(int** &matrix)
    {
        for (int i = 1; i < this->vertices; i++)
            for (int j = 0; j < i; j++)
                if (matrix[i][j] != 0)
                    matrix[j][i] = matrix[i][j];
        for (int i = 0; i < this->vertices - 1; i++)
            for (int j = i+1; j < this->vertices; j++)
                if (matrix[i][j] != 0)
                    matrix[j][i] = matrix[i][j];
    }

    void fix_vertex(int v)
    {
        if (v == 0)
            this->matrix[1][0] = random_num(1, 100);
        else
            this->matrix[v][0] = random_num(1, 100);
        this->edges++;
    }

    void random_fix()
    {
        for (int i = 1; i < this->vertices; i++)
            for (int j = 0; j < i; j++)
                if (this->matrix[i][j] == 0)
                {
                    this->matrix[i][j] = random_num(1, 100);
                    this->edges++;
                    return;
                }
    }

    void count_edges()
    {
        int edges = 0;
        for (int i = 1; i < this->vertices; i++)
            for (int j = 0; j < i; j++)
                if (this->matrix[i][j] != 0)
                    edges++;
        this->edges = edges;
    }

    void is_connected(bool* visited, int v = 0)
    {
        visited[v] = true;
        for (int i = 0; i < this->vertices; i++)
        {
            if (this->matrix[v][i] != 0 && !visited[i])
                is_connected(visited, i);
        }
    }

    void all_visited_in(bool* visited)
    {
        for (int i = 0; i < this->vertices; i++)
            if (!visited[i])
            {
                this->matrix[0][i] = random_num(1,100);
                this->matrix[i][0] = this->matrix[0][i];
                is_connected(visited, 0);
            }
    }

    void is_connected_control()
    {
        bool* visited = new bool[this->vertices];
        fill_false(visited, this->vertices);
        is_connected(visited, 0);
        all_visited_in(visited);

        delete [] visited;
    }

    void check_completness()
    {
        this->is_connected_control();
        this->count_edges();
    }

    std::vector<int> get_neighbours(int v)
    {
        std::vector<int> N;
        for (int i = 0; i < this->vertices; i++)
            if (this->matrix[v][i] != 0)
                N.push_back(i);
        return N;
    }

    std::vector<int> filter_out(std::vector<int> Q, std::vector<int> N)
    {
        std::vector<int> W;
        for (int x : Q)
            for (int y : N)
                if (x == y)
                    W.push_back(x);
        return W;
    }

public:
    Graph(int v)
    {
        this->matrix = create_matrix(v);
        fill_zeros(this->matrix, v);
        this->vertices = v;
        this->generate_directed_graph();
        this->convert_to_undirected();
        this->check_completness();
    }

    void convert_to_matrix(std::vector<int> ls, int** &matrix)
    {
        for (int i = 0; i < ls.size() - 1; i++)
            matrix[ls[i]][ls[i+1]] = 1;
        this->fix_matrix(matrix);
    }

    void convert_to_adj(std::vector<int> road)
    {
        int** matrix = create_matrix(this->vertices);
        fill_zeros(matrix, this->vertices);
        this->convert_to_matrix(road, matrix);

        for (int i = 1; i < this->vertices; i++)
            for (int j = 0; j < i; j++)
                if (this->matrix[i][j])
                {
                    Edge edge;
                    edge.a = i;
                    edge.b = j;
                    edge.w = this->matrix[i][j];
                    edge.c = matrix[i][j] ? 'r' : 'b';
                    this->adj.push_back(edge);
                }
    }

    void show_matrix()
    {
        std::cout << "\t";
        for (int i = 0; i < this->vertices; i++)
            std::cout << i << "\t";
        std::cout << std::endl;
        for (int i = 0; i < this->vertices; i++)
        {
            std::cout << i << "\t";
            for (int j = 0; j < this->vertices; j++)
                std::cout << this->matrix[i][j] << "\t";
            std::cout << std::endl;
        }
    }

    void show_adj()
    {
        for (Edge e : this->adj)
            std::cout << e.a << " " << e.b << " " << e.w << " " << e.c << std::endl;
    }

    int min(std::vector<int> vec, std::vector<bool> Q)
    {
        int min = INT_MAX;
        int min_i = 0;
        for (int i = 0; i < vec.size(); i++)
        {
            if (vec[i] < min && !Q[i])
            {
                min = vec[i];
                min_i = i;
            }
        }
        return min_i;
    }

    int find_in_vector(std::vector<int> vec, int s)
    {
        for (int i = 0; i < vec.size(); i++)
        {
            if (vec[i] == s)
                return i;
        }
        return -1;
    }

    bool all_true(std::vector<bool> vec)
    {
        for (bool b : vec)
            if (!b)
                return false;
        return true;
    }

    std::vector<int> get_neighbours_still_not_visited(int v, std::vector<bool> Q)
    {
        std::vector<int> N;
        for (int i = 0; i < this->vertices; i++)
            if (this->matrix[v][i] != 0 && !Q[i])
                N.push_back(i);
        return N;
    }

    std::vector<int> find_shortest_path(int from, int to)
    {
        std::vector<int> dist(this->vertices, INT_MAX);
        std::vector<int> prev(this->vertices, -1);
        std::vector<bool> visited(this->vertices, false);

        dist[from] = 0;
        while (!all_true(visited))
        {
            int u = min(dist, visited);
            if (u == to)
                break;
            visited[u] = true;
            for (int v : get_neighbours_still_not_visited(u, visited))
            {
                int alt = dist[u] + this->matrix[u][v];
                if (alt < dist[v])
                {
                    dist[v] = alt;
                    prev[v] = u;
                }
            }
        }
        std::vector<int> road;
        int u = to;
        if (prev[u] != -1 || u == from)
        {
            while (u != -1)
            {
                road.push_back(u);
                u = prev[u];
            }
        }
        return road;
    }

    std::vector<int> find_shortest_path_openmp(int from, int to)
    {
        std::vector<int> dist(this->vertices, INT_MAX);
        std::vector<int> prev(this->vertices, -1);
        std::vector<bool> visited(this->vertices, false);

        dist[from] = 0;
        while (!all_true(visited))
        {
            int u = min(dist, visited);
            if (u == to)
                break;
            visited[u] = true;
            auto neighbours = get_neighbours_still_not_visited(u, visited);

            #pragma omp parallel for shared(dist, prev, neighbours, visited, u)
            for (int i = 0; i < neighbours.size(); i++)
            {
                int alt = dist[u] + this->matrix[u][neighbours[i]];
                #pragma omp critical
                {
                    if (alt < dist[neighbours[i]])
                    {
                        dist[neighbours[i]] = alt;
                        prev[neighbours[i]] = u;
                    }
                }
            }
        }
        std::vector<int> road;
        int u = to;
        if (prev[u] != -1 || u == from)
        {
            while (u != -1)
            {
                road.push_back(u);
                u = prev[u];
            }
        }
        return road;
    }
};
