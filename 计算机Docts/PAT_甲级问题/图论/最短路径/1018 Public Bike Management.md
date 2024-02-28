## 1018 Public Bike Management (30 分)

There is a public bike service in Hangzhou City which provides great convenience to the tourists from all over the world. One may rent a bike at any station and return it to any other stations in the city.

The Public Bike Management Center (PBMC) keeps monitoring the real-time capacity of all the stations. A station is said to be in **perfect** condition if it is exactly half-full. If a station is full or empty, PBMC will collect or send bikes to adjust the condition of that station to perfect. And more, all the stations on the way will be adjusted as well.

When a problem station is reported, PBMC will always choose the shortest path to reach that station. If there are more than one shortest path, the one that requires the least number of bikes sent from PBMC will be chosen.

![img](https://images.ptausercontent.com/213)

The above figure illustrates an example. The stations are represented by vertices and the roads correspond to the edges. The number on an edge is the time taken to reach one end station from another. The number written inside a vertex *S* is the current number of bikes stored at *S*. Given that the maximum capacity of each station is 10. To solve the problem at *S*3, we have 2 different shortest paths:

1. PBMC -> *S*1 -> *S*3. In this case, 4 bikes must be sent from PBMC, because we can collect 1 bike from *S*1 and then take 5 bikes to *S*3, so that both stations will be in perfect conditions.
2. PBMC -> *S*2 -> *S*3. This path requires the same time as path 1, but only 3 bikes sent from PBMC and hence is the one that will be chosen.

### Input Specification:

Each input file contains one test case. For each case, the first line contains 4 numbers: *C**ma**x* (≤100), always an even number, is the maximum capacity of each station; *N* (≤500), the total number of stations; *S**p*, the index of the problem station (the stations are numbered from 1 to *N*, and PBMC is represented by the vertex 0); and *M*, the number of roads. The second line contains *N* non-negative numbers *C**i* (*i*=1,⋯,*N*) where each *C**i* is the current number of bikes at *S**i* respectively. Then *M* lines follow, each contains 3 numbers: *S**i*, *S**j*, and *T**ij* which describe the time *T**ij* taken to move betwen stations *S**i* and *S**j*. All the numbers in a line are separated by a space.

### Output Specification:

For each test case, print your results in one line. First output the number of bikes that PBMC must send. Then after one space, output the path in the format: 0−>*S*1−>⋯−>*S**p*. Finally after another space, output the number of bikes that we must take back to PBMC after the condition of *S**p* is adjusted to perfect.

Note that if such a path is not unique, output the one that requires minimum number of bikes that we must take back to PBMC. The judge's data guarantee that such a path is unique.

### Sample Input:

```in
10 3 3 5
6 7 0
0 1 1
0 2 1
0 3 3
1 3 1
2 3 1结尾无空行
```

### Sample Output:

```out
3 0->2->3 0
```

```C++
/*Dijkstra+DFS Dijkstra求最短路径，DFS遍历所有最短路径，通过比较最小的send和back更新路径*/
/*注意点：：只能沿着最短路径的方向收集多余自行车，分给后面的节点，
后面节点多出来的不能填到前面去，只能计入回收总量，例如路径上自行车数为5->0->10，
并不能把最后一个节点上挪5个给中间的，需要送出5个，并回收5个。
所以总需求量不能用Cmax / 2 * 节点数 - 现有数来计算。*/

#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
const int MAX = 10000;
struct Edge{
    int to;
    int length;
    Edge() : to(), length(){};
    Edge(int t, int l) : to(t), length(l){};
};
vector<int> bikes;
vector<Edge> Graph[510];
vector<int> preNode[510]; //前驱结点
vector<int> visited;
vector<int> dst;
vector<int> path, temp_path;
int send=MAX, back=MAX;
int capacity, N, Sp, M;
//由于是记录preNode，所以从终点向起点遍历，前驱结点记录结构使图成为树结构，所以不需要visited辅助数组
void DFS(int index){
    //将结点index压入路径
    temp_path.push_back(index);
    if(index==0){
        int temp_send = 0, temp_back = 0;
        for (int i = temp_path.size()-1; i>=0;i--){  //从0点开始模拟路径
            int node = temp_path[i];
            int need = capacity / 2 - bikes[node];
            if(need<0)
                temp_back += abs(need);
            else if(need>0){
                if(temp_back-need>=0){
                    temp_back -= need;
                }else{
                    temp_send += abs(need) - temp_back;
                    temp_back = 0;
                }
            }
        }
        if(temp_send<send){
            send = temp_send;
            back = temp_back;
            path = temp_path;
        }
        else if(temp_send==send&&temp_back<back){
            back = temp_back;
            path = temp_path;
        } 
    }
    for (int i = 0; i < preNode[index].size();i++){
        DFS(preNode[index][i]);
    }
    //回溯 使得可以DFS遍历所有路径
    temp_path.pop_back();
}

int main(){
    cin >> capacity >> N >> Sp >> M;
    visited = vector<int>(N + 1, 0);
    dst = vector<int>(N + 1, MAX);

    bikes.push_back(capacity/2);
    int temp;
    for (int i = 1; i <=N;i++){
        cin >> temp;
        bikes.push_back(temp);
    }
    int from, to, length;
    for (int i = 0; i < M;i++){
        cin >> from >> to >> length;
        Graph[from].push_back(Edge(to, length));
        Graph[to].push_back(Edge(from, length));
    }
    //初始化dst[0]
    dst[0] = 0;
    for (int i = 0; i <= N;i++){
        int k = -1, min = MAX;
        for (int j = 0; j <= N;j++){
            if(dst[j]<min&&visited[j]==0){
                k = j;
                min = dst[j];
            }
        }
        if(k==-1||k==Sp)
            break;
        visited[k] = 1;
        //更新点k的后继结点
        for (int j = 0; j < Graph[k].size();j++){
            int node = Graph[k][j].to;
            if(visited[node]==0){
                if(dst[k]+Graph[k][j].length<dst[node]){
                    dst[node] = dst[k] + Graph[k][j].length;
                    preNode[node].clear();
                    preNode[node].push_back(k);
                }
                else if (dst[k] + Graph[k][j].length == dst[node]){
                    preNode[node].push_back(k);
                }
            }
        }
    }
    DFS(Sp);
    if(send==MAX)
        send = 0;
    if(back==MAX)
        back = 0;
    cout << send<<" ";
    for (int i = path.size() - 1; i > 0;i--){
        cout << path[i] << "->";
    }
    cout << Sp << " " << back;
}
```

