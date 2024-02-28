## 1003 Emergency (25 分)

As an emergency rescue team leader of a city, you are given a special map of your country. The map shows several scattered cities connected by some roads. Amount of rescue teams in each city and the length of each road between any pair of cities are marked on the map. When there is an emergency call to you from some other city, your job is to lead your men to the place as quickly as possible, and at the mean time, call up as many hands on the way as possible.

### Input Specification:

Each input file contains one test case. For each test case, the first line contains 4 positive integers: *N* (≤500) - the number of cities (and the cities are numbered from 0 to *N*−1), *M* - the number of roads, *C*1 and *C*2 - the cities that you are currently in and that you must save, respectively. The next line contains *N* integers, where the *i*-th integer is the number of rescue teams in the *i*-th city. Then *M* lines follow, each describes a road with three integers *c*1, *c*2 and *L*, which are the pair of cities connected by a road and the length of that road, respectively. It is guaranteed that there exists at least one path from *C*1 to *C*2.

### Output Specification:

For each test case, print in one line two numbers: the number of different shortest paths between *C*1 and *C*2, and the maximum amount of rescue teams you can possibly gather. All the numbers in a line must be separated by exactly one space, and there is no extra space allowed at the end of a line.

### Sample Input:

```in
5 6 0 2
1 2 1 5 3
0 1 1
0 2 2
0 3 1
1 2 1
2 4 1
3 4 1
```

### Sample Output:

```out
2 4
```

```C++
#include<cstdio>
#include<iostream>
#include<queue>
#include<vector>
#include<string>
using namespace std;   /*Dijkstra---邻接表*/
const int MAX = 1000;
struct Edge{
    int to;//指向的点
    int length;//边的长度
    Edge():to(),length(){} //无参构造函数
    Edge(int t,int l):to(t),length(l){}  //有参构造函数
};
vector<Edge> Graph[500];
vector<int> visited;
vector<int> paths; //起点到结点最短距离的数量
vector<int> dst; //起点到当前点的距离
vector<int> teams; //每个结点的队伍数
vector<int> gathered;//每个结点能聚集到的队伍
int N, M, C1, C2;

int main(){
    cin >> N >> M >> C1 >> C2;
    //初始化
    visited = vector<int>(N, 0);
    dst = vector<int>(N, MAX);
    paths=vector<int>(N, 0);
    int num;
    for (int i = 0; i < N;i++){
        cin >> num;
        teams.push_back(num);
    }
    gathered = teams; //拷贝初始化：gathered与teams相同 或 gathered=vector<int>(teams);
    int from, to, length;
    for (int i = 0; i < M;i++){
        cin >> from >> to >> length;
        Graph[from].push_back(Edge(to, length));
        Graph[to].push_back(Edge(from, length));
    }
    dst[C1] = 0;
    paths[C1] = 1;//起点的路径初始化为1条
    //Dijkstra
    for (int i = 0; i < N;i++){
        int k=-1;int min=MAX; //k:最近点的标号 min：最近点的距离
        for (int j = 0; j < N;j++){
            //找到dst中最距离起点最近的一个结点
            if(visited[j]==0&&dst[j]<min){
                k = j;
                min = dst[j];
            }
        }
        //找到k,min
        if(k==-1||k==C2)  //找到终点或没有可选点，退出循环
            break;
        //松弛操作：更新dst中 k点相邻的未访问的结点
        visited[k] = 1;
        for (int j = 0; j < Graph[k].size();j++){
            int node = Graph[k][j].to;
            if(visited[node]==0&&Graph[k][j].length+dst[k]<=dst[node]){
                if(Graph[k][j].length+dst[k]<dst[node]){  
                    dst[node] = Graph[k][j].length + dst[k];
                    paths[node] = paths[k];//C1到node的数量更新为k的数量
                    gathered[node] =gathered[k]+teams[node]; //注意这里只要是更短的距离就要更新gathered
                }
                //若找到相等的最短路径，C1到node的数量更新为C1到k的数量+原来的C1-node数量
                else{
                    paths[node] += paths[k];
                    if(gathered[node]<gathered[k]+teams[node])
                        //保存聚集的队伍数量
                        gathered[node] =gathered[k]+teams[node];
                }
            }
        }
    }
    cout << paths[C2] << " " << gathered[C2];
}

```

