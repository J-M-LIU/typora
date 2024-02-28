## 1076 Forwards on Weibo (30 分)

Weibo is known as the Chinese version of Twitter. One user on Weibo may have many followers, and may follow many other users as well. Hence a social network is formed with followers relations. When a user makes a post on Weibo, all his/her followers can view and forward his/her post, which can then be forwarded again by their followers. Now given a social network, you are supposed to calculate the maximum potential amount of forwards for any specific user, assuming that only *L* levels of indirect followers are counted.

### Input Specification:

Each input file contains one test case. For each case, the first line contains 2 positive integers: *N* (≤1000), the number of users; and *L* (≤6), the number of levels of indirect followers that are counted. Hence it is assumed that all the users are numbered from 1 to *N*. Then *N* lines follow, each in the format:

```
M[i] user_list[i]
```

where `M[i]` (≤100) is the total number of people that `user[i]` follows; and `user_list[i]` is a list of the `M[i]` users that followed by `user[i]`. It is guaranteed that no one can follow oneself. All the numbers are separated by a space.

Then finally a positive *K* is given, followed by *K* `UserID`'s for query.

### Output Specification:

For each `UserID`, you are supposed to print in one line the maximum potential amount of forwards this user can trigger, assuming that everyone who can view the initial post will forward it once, and that only *L* levels of indirect followers are counted.

### Sample Input:

```in
7 3
3 2 3 4
0
2 5 6
2 3 1
2 3 4
1 4
1 5
2 2 6
```

### Sample Output:

```out
4
5
```

#### 思路：

1. 注意本题的社交网络中，强调层级关系，似树状结构，如果用DFS，没有解决同层的结点间的联系
2. 只能用BFS
3. 注意visited[]辅助数组的置1位置，while循环外第一个结点visited置1，while内判断弹出结点的子结点，入队时visited置1



```C++
//问题：本题中需要注意，如果使用DFS，会存在重复计算的问题，因为该题不是树状结构，同一层的结点存在联系，无法计算
//故使用BFS
#include<cstdio>
#include<iostream>
#include<vector>
#include<queue>
using namespace std;

vector<int>Graph[1010];
int node_level[1010];
vector<int>visited;
int N,L;
int BFS(int index){
    queue<int> Q;
    Q.push(index);
    visited[index]=1;
    int count=0;
    while(!Q.empty()){
        int user=Q.front();
        Q.pop();
        if(node_level[user]>L) break;
        count++;
        for(int i=0;i<Graph[user].size();i++){
            int id=Graph[user][i];
            if(visited[id]==0){
                node_level[id]=node_level[user]+1;
                Q.push(id);
                visited[id] = 1;
            }
        }
    }
    return count-1;
}
int main(){
    cin>>N>>L;
    visited=vector<int>(N+1,0);
    int follow,M;
    for(int i=1;i<=N;i++){
        cin>>M;
        for(int j=0;j<M;j++){
            cin>>follow;
            Graph[follow].push_back(i);
        }
    }
    int query;
    cin>>M;
    for(int i=1;i<=M;i++){
        cin>>query;
        int count=BFS(query);
        cout<<count<<endl;
        fill(visited.begin(), visited.end(), 0);
        fill(node_level,node_level+N+1,0);
    }
    return 0;
}
```

