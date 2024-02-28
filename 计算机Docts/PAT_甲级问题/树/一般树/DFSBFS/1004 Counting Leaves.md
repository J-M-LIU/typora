## 1004 Counting Leaves (30 分)

A family hierarchy is usually presented by a pedigree tree. Your job is to count those family members who have no child.

### Input Specification:

Each input file contains one test case. Each case starts with a line containing 0<*N*<100, the number of nodes in a tree, and *M* (<*N*), the number of non-leaf nodes. Then *M* lines follow, each in the format:

```
ID K ID[1] ID[2] ... ID[K]
```

where `ID` is a two-digit number representing a given non-leaf node, `K` is the number of its children, followed by a sequence of two-digit `ID`'s of its children. For the sake of simplicity, let us fix the root ID to be `01`.

The input ends with *N* being 0. That case must NOT be processed.

### Output Specification:

For each test case, you are supposed to count those family members who have no child **for every seniority level** starting from the root. The numbers must be printed in a line, separated by a space, and there must be no extra space at the end of each line.

The sample case represents a tree with only 2 nodes, where `01` is the root and `02` is its only child. Hence on the root `01` level, there is `0` leaf node; and on the next level, there is `1` leaf node. Then we should output `0 1` in a line.

### Sample Input:

```in
2 1
01 1 02
```

### Sample Output:

```out
0 1
```



#### DFS做法

```C++
#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
vector<int> Graph[101];
vector<int> Level(101,0);
int N, M;
int max_level = 0;

void DFS(int index,int level){
    if(Graph[index].size()==0){ //是叶子结点
        max_level = max(max_level, level);
        Level[level]++;
        return;
    }
    for (int i = 0; i < Graph[index].size();i++){
        DFS(Graph[index][i], level + 1);
    }
}
int main(){    //root是第0层，从低到高
    cin >> N >> M;
    int parent, K, id;
    //初始化
    for (int i = 0; i < M;i++){
        cin >> parent >> K;
        for (int j = 0; j < K;j++){
            cin >> id;
            Graph[parent].push_back(id);
        }
    }
    DFS(1, 0);
    for (int i = 0; i < max_level;i++){
        cout << Level[i] << " ";
    }
    cout << Level[max_level];
}
```

#### BFS做法

```C++
#include<cstdio>
#include<iostream>
#include<vector>
#include<queue>
using namespace std;

vector<int> Graph[101];
vector<int> Level(101,0);
vector<int> node_level(101, 0);    //记录每一个结点的层数
queue<int>q;
int N, M;
int max_level = 0;

void BFS(){
    q.push(1);
    int k;
    while(!q.empty()){
        k = q.front();
        q.pop();
        if(Graph[k].size()==0){
            //说明是叶子结点
            Level[node_level[k]]++;
        }
        max_level = node_level[k];
        for (int i = 0; i < Graph[k].size();i++){
            node_level[Graph[k][i]]=node_level[k]+1; //孩子节点的层数=父亲结点层数+1
            q.push(Graph[k][i]);
        }
    }
}
int main(){    //root是第0层，从低到高
    cin >> N >> M;
    int parent, K, id;
    //初始化
    for (int i = 0; i < M;i++){
        cin >> parent >> K;
        for (int j = 0; j < K;j++){
            cin >> id;
            Graph[parent].push_back(id);
        }
    }
    BFS();
    for (int i = 0; i < max_level;i++){
        cout << Level[i] << " ";
    }
    cout << Level[max_level];
}
```

