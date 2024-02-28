## 1030 Travel Plan (30 分)

A traveler's map gives the distances between cities along the highways, together with the cost of each highway. Now you are supposed to write a program to help a traveler to decide the shortest path between his/her starting city and the destination. If such a shortest path is not unique, you are supposed to output the one with the minimum cost, which is guaranteed to be unique.

### Input Specification:

Each input file contains one test case. Each case starts with a line containing 4 positive integers *N*, *M*, *S*, and *D*, where *N* (≤500) is the number of cities (and hence the cities are numbered from 0 to *N*−1); *M* is the number of highways; *S* and *D* are the starting and the destination cities, respectively. Then *M* lines follow, each provides the information of a highway, in the format:

```
City1 City2 Distance Cost
```

where the numbers are all integers no more than 500, and are separated by a space.

### Output Specification:

For each test case, print in one line the cities along the shortest path from the starting point to the destination, followed by the total distance and the total cost of the path. The numbers must be separated by a space and there must be no extra space at the end of output.

### Sample Input:

```in
4 5 0 3
0 1 1 20
1 3 2 30
0 3 4 10
0 2 2 20
2 3 1 20
```

### Sample Output:

```out
0 2 3 3 40
```

```C++
#include <algorithm>
#include <cstdio>
#include <iostream>
#include<vector>
using namespace std;
const int MAX = 1000;
struct Edge{
    int to;int length;int cost;
    Edge(int t, int l, int c) : to(t), length(l), cost(c){};
};
vector<Edge> Graph[510];
vector<int> visited;
vector<int> dst;
vector<int> preNode; //每个结点的前驱结点
vector<int> cost; //每个结点的累积cost值
vector<int> path; //起点到终点的路径
int N, M, S, D;
void DFS(int index){
    path.push_back(index);
    if(index!=S){
        DFS(preNode[index]);
    }
}
int main(){
    cin >> N >> M >> S >> D;
    int from, to, l, c;
    visited = vector<int>(N, 0);
    dst = vector<int>(N, MAX);
    cost = vector<int>(N, 0);
    preNode = vector<int>(N, 0);
    for (int i = 0; i < M;i++){
        cin >> from >> to >> l >> c;
        Graph[from].push_back(Edge(to, l, c));
        Graph[to].push_back(Edge(from, l, c));
    }
    dst[S] = 0;
    for (int i = 0; i < N;i++){
        int k = -1, min = MAX;
        for (int j = 0; j < N;j++){
            if(dst[j]<min&&visited[j]==0){
                k = j;
                min = dst[j];
            }
        }
        if(k==-1||k==D)
            break;
        visited[k] = 1;
        for (int j = 0; j < Graph[k].size();j++){
            int node = Graph[k][j].to;
            if(visited[node]==0){
                if (Graph[k][j].length + dst[k] < dst[node]){
                    preNode[node] = k;
                    cost[node] = cost[k] + Graph[k][j].cost;
                    dst[node] = Graph[k][j].length + dst[k];
                }
                else if (Graph[k][j].length + dst[k]==dst[node]){
                    if(cost[k]+Graph[k][j].cost<cost[node]){
                        cost[node] = cost[k] + Graph[k][j].cost;
                        preNode[node] = k;
                    }
                }
            }
        }
    }
    DFS(D);
    for (int i = path.size() - 1; i >= 0;i--)
        cout << path[i] << " ";
    cout << dst[D] << " " << cost[D];
}
```

