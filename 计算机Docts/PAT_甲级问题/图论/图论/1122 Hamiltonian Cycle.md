## 1122 Hamiltonian Cycle (25 分)

The "Hamilton cycle problem" is to find a simple cycle that contains every vertex in a graph. Such a cycle is called a "Hamiltonian cycle".

In this problem, you are supposed to tell if a given cycle is a Hamiltonian cycle.

### Input Specification:

Each input file contains one test case. For each case, the first line contains 2 positive integers *N* (2<*N*≤200), the number of vertices, and *M*, the number of edges in an undirected graph. Then *M* lines follow, each describes an edge in the format `Vertex1 Vertex2`, where the vertices are numbered from 1 to *N*. The next line gives a positive integer *K* which is the number of queries, followed by *K* lines of queries, each in the format:

*n* *V*1 *V*2 ... *V**n*

where *n* is the number of vertices in the list, and *V**i*'s are the vertices on a path.

### Output Specification:

For each query, print in a line `YES` if the path does form a Hamiltonian cycle, or `NO` if not.

### Sample Input:

```in
6 10
6 2
3 4
1 5
2 5
3 1
4 1
1 6
6 3
1 2
4 5
6
7 5 1 4 3 6 2 5
6 5 1 4 3 6 2
9 6 2 1 6 3 4 5 2 6
4 1 2 5 1
7 6 1 3 4 5 2 6
7 6 1 2 5 4 3 1
```

### Sample Output:

```out
YES
NO
NO
NO
YES
NO
```

**不能判定为真的情况：**

1. 首尾结点不一致
2. 出现多个环（如果当前点x visited[x]=1,则存在一个环）
3. 访问的结点数<n

```C++
#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
int Graph[210][210];
int visited[210];
vector<int> temp;
int n,m,k;
int main(){
    cin>>n>>m;
    int s1,s2;
    for(int i=0;i<m;i++){
        cin>>s1>>s2;
        Graph[s1][s2]=Graph[s2][s1]=1;
    }
    cin>>k;
    int num,id;
    for(int i=0;i<k;i++){
        cin>>num;
        for(int j=0;j<num;j++){
            cin>>id;
            temp.push_back(id);
        }
        bool flag=true;
        int cnt=1;
        int cycle=0;
        visited[temp[0]]=1;
        for(int j=1;j<num;j++){
            if(Graph[temp[j-1]][temp[j]]==1){
                if(visited[temp[j]]==0){
                    cnt++;visited[temp[j]]=1;
                }
                else{
                    if(cycle>0){flag=false;break;}
                    cycle++;
                }
                
            }
            else{
                flag = false;break;
            }
        }
        if(cnt!=n||cycle==0||temp[0]!=temp[num-1]) flag=false;
        printf(flag?"YES\n":"NO\n");
        temp.clear();
        fill(visited,visited+n+1,0);
    }
}
```

