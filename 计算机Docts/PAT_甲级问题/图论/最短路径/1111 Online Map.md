## 1111 Online Map (30 分)

Input our current position and a destination, an online map can recommend several paths. Now your job is to recommend two paths to your user: one is the shortest, and the other is the fastest. It is guaranteed that a path exists for any request.

### Input Specification:

Each input file contains one test case. For each case, the first line gives two positive integers *N* (2≤*N*≤500), and *M*, being the total number of streets intersections on a map, and the number of streets, respectively. Then *M* lines follow, each describes a street in the format:

```
V1 V2 one-way length time
```

where `V1` and `V2` are the indices (from 0 to *N*−1) of the two ends of the street; `one-way` is 1 if the street is one-way from `V1` to `V2`, or 0 if not; `length` is the length of the street; and `time` is the time taken to pass the street.

Finally a pair of source and destination is given.

### Output Specification:

For each case, first print the shortest path from the source to the destination with distance `D` in the format:

```
Distance = D: source -> v1 -> ... -> destination
```

Then in the next line print the fastest path with total time `T`:

```
Time = T: source -> w1 -> ... -> destination
```

In case the shortest path is not unique, output the fastest one among the shortest paths, which is guaranteed to be unique. In case the fastest path is not unique, output the one that passes through the fewest intersections, which is guaranteed to be unique.

In case the shortest and the fastest paths are identical, print them in one line in the format:

```
Distance = D; Time = T: source -> u1 -> ... -> destination
```

### Sample Input 1:

```in
10 15
0 1 0 1 1
8 0 0 1 1
4 8 1 1 1
3 4 0 3 2
3 9 1 4 1
0 6 0 1 1
7 5 1 2 1
8 5 1 2 1
2 3 0 2 2
2 1 1 1 1
1 3 0 3 1
1 4 0 1 1
9 7 1 3 1
5 1 0 5 2
6 5 1 1 2
3 5
```

### Sample Output 1:

```out
Distance = 6: 3 -> 4 -> 8 -> 5
Time = 3: 3 -> 1 -> 5
```

### Sample Input 2:

```in
7 9
0 4 1 1 1
1 6 1 1 3
2 6 1 1 1
2 5 1 2 2
3 0 0 1 1
3 1 1 1 3
3 2 1 1 2
4 5 0 2 2
6 5 1 1 2
3 5
```

### Sample Output 2:

```out
Distance = 3; Time = 4: 3 -> 2 -> 5
```

**注意的问题：**

1. 第一条路径判断最短路径，若不唯一，则取时间最短
2. 第二条路径判断时间最短，若不唯一，取访问结点最少
3. 我只用了Dijkstra，也可以Dijkstra+DFS解题

```C++
#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
struct Edge{
    int to;int length;int time;
};
const int MAX=999999;
vector<Edge>Graph[510];
int visited[510];
int dst[510];
int times[510];
int dst_pre[510],time_pre[510];
int dst_cnt[510],time_cnt[510];
int n,m;
int s1,s2;
int main(){
    cin>>n>>m;
    int from,to,length,time,one_way;
    for(int i=0;i<m;i++){
        cin>>from>>to>>one_way>>length>>time;
        Edge e;e.to=to;e.length=length;e.time=time;
        Graph[from].push_back(e);
        if(one_way==0){e.to=from;Graph[to].push_back(e);}
    }
    cin>>s1>>s2;
    //distance
    fill(dst,dst+n,MAX);dst[s1]=0;
    for(int i=0;i<n;i++){
        int k=-1,min=MAX;
        for(int j=0;j<n;j++){
            if(visited[j]==0&&dst[j]<min){
                min=dst[j];k=j;
            }
        }
        if(k==-1||k==s2) break;
        visited[k]=1;
        for(int j=0;j<Graph[k].size();j++){
            int node=Graph[k][j].to;
            if(visited[node]==0){
                if(dst[k]+Graph[k][j].length<dst[node]){
                    dst[node] = dst[k] + Graph[k][j].length;
                    dst_pre[node]=k;
                    dst_cnt[node]=dst_cnt[k]+Graph[k][j].time;
                }
                else if(dst[k]+Graph[k][j].length==dst[node]&&dst_cnt[k]+Graph[k][j].time<dst_cnt[node]){
                    dst_pre[node] = k;
                    dst_cnt[node] = dst_cnt[k] + Graph[k][j].time;
                }
            }
        }
    }
    fill(times,times+n,MAX); times[s1]=0;
    fill(visited,visited+n,0);
    for(int i=0;i<n;i++){
        int k=-1,min=MAX;
        for(int j=0;j<n;j++){
            if (visited[j] == 0 &&times[j] < min) {
                min=times[j];k=j;
            }
        }
        if(k==-1||k==s2)break;
        visited[k]=1;
        for(int j=0;j<Graph[k].size();j++){
            int node=Graph[k][j].to;
            if(visited[node]==0){
                if(times[k]+Graph[k][j].time<times[node]){
                    times[node] = times[k] + Graph[k][j].time;
                    time_pre[node] = k;
                    time_cnt[node]=time_cnt[k]+1;
                }
                else if(times[k]+Graph[k][j].time==times[node]&&time_cnt[k]+1<time_cnt[node]){
                    time_pre[node] = k;
                    time_cnt[node]=time_cnt[k]+1;
                }
            }
        }
    }
    vector<int>temp,temp2;
    int node=s2; 
    while(node!=s1){
        temp.push_back(node);
        node=dst_pre[node];
    }
    node=s2;
    while (node != s1) {
        temp2.push_back(node);
        node = time_pre[node];
    }
    bool flag=true;
    for(int i=0;i<temp.size()&&i<temp2.size();i++){
        if(temp[i]!=temp2[i]){flag=false;break;}
    }
    if(flag)
    printf("Distance = %d; ",dst[s2]);
    else{
        printf("Distance = %d: %d",dst[s2],s1);
        for(int i=temp.size()-1;i>=0;i--){
        printf(" -> %d",temp[i]);
    }
    }
    printf(flag?"":"\n");
    printf("Time = %d: %d", times[s2], s1);
    for (int i = temp2.size() - 1; i >= 0; i--) {
        printf(" -> %d", temp2[i]);
    }
}
```

