## 1013 Battle Over Cities (25 分)

It is vitally important to have all the cities connected by highways in a war. If a city is occupied by the enemy, all the highways from/toward that city are closed. We must know immediately if we need to repair any other highways to keep the rest of the cities connected. Given the map of cities which have all the remaining highways marked, you are supposed to tell the number of highways need to be repaired, quickly.

For example, if we have 3 cities and 2 highways connecting *c**i**t**y*1-*c**i**t**y*2 and *c**i**t**y*1-*c**i**t**y*3. Then if *c**i**t**y*1 is occupied by the enemy, we must have 1 highway repaired, that is the highway *c**i**t**y*2-*c**i**t**y*3.

### Input Specification:

Each input file contains one test case. Each case starts with a line containing 3 numbers *N* (<1000), *M* and *K*, which are the total number of cities, the number of remaining highways, and the number of cities to be checked, respectively. Then *M* lines follow, each describes a highway by 2 integers, which are the numbers of the cities the highway connects. The cities are numbered from 1 to *N*. Finally there is a line containing *K* numbers, which represent the cities we concern.

### Output Specification:

For each of the *K* cities, output in a line the number of highways need to be repaired if that city is lost.

### Sample Input:

```in
3 2 3
1 2
1 3
1 2 3
```

### Sample Output:

```out
1
0
0
```

```C++
#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
vector<int> Graph[1001];
vector<int> visited;
int N, M, K;
void DFS(int node){
    visited[node] = 1;
    for (int i = 0; i < Graph[node].size();i++){
    if(visited[Graph[node][i]]==0)
        DFS(Graph[node][i]);
    }
}
int main(){
    cin >> N >> M >> K;
    int from,to;
    for (int i = 0; i < M;i++){
        cin >> from >> to;
        Graph[from].push_back(to);   //注意这里是双向的
        Graph[to].push_back(from);
    }
    //若点1被占领，剩下的点构成的连通分量的数目-1为需要修复的道路数
    int node, count;
    for (int i = 0; i < K;i++){
        count = 0;
        cin >> node;
        visited = vector<int>(N+1, 0);
        visited[node] = 1;
        for (int j = 1; j <= N;j++){
            if(visited[j]==0){
                count++;
                DFS(j);
            }
        }
        cout << count-1;
        if(i!=K-1)
            printf("\n");
    }
    return 0;
}
```

