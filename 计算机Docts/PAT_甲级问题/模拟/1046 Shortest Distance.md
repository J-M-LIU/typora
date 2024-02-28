## 1046 Shortest Distance (20 分)

The task is really simple: given *N* exits on a highway which forms a simple cycle, you are supposed to tell the shortest distance between any pair of exits.

### Input Specification:

Each input file contains one test case. For each case, the first line contains an integer *N* (in [3,105]), followed by *N* integer distances *D*1 *D*2 ⋯ *D**N*, where *D**i* is the distance between the *i*-th and the (*i*+1)-st exits, and *D**N* is between the *N*-th and the 1st exits. All the numbers in a line are separated by a space. The second line gives a positive integer *M* (≤104), with *M* lines follow, each contains a pair of exit numbers, provided that the exits are numbered from 1 to *N*. It is guaranteed that the total round trip distance is no more than 107.

### Output Specification:

For each test case, print your results in *M* lines, each contains the shortest distance between the corresponding given pair of exits.

### Sample Input:

```in
5 1 2 4 14 9
3
1 3
2 5
4 1
```

### Sample Output:

```out
3
10
7
```

**arr[i]存放第一个点到点i的下一点的距离：如arr[3],存放1到4的距离，arr[n]则是环形路的总长度和sum**

**点a到b的距离有两个方向，即arr[b-1]-arr[a-1] 和 sum-abs(arr[b-1]-arr[a-1])**

```C++
#include<cstdio>
#include<iostream>
using namespace std;
int n;
int arr[100010];
int main(){
    cin>>n;
    int dis,sum;
    for(int i=1;i<=n;i++){
        cin>>arr[i];
        arr[i]+=arr[i-1];
    }
    sum=arr[n];
    int k,s1,s2;
    cin>>k;
    for(int i=0;i<k;i++){
        cin>>s1>>s2;
        dis=min(abs(arr[s2-1]-arr[s1-1]),sum-abs(arr[s2-1]-arr[s1-1]));
        printf("%d\n",dis);
    }   
    return 0;
}
```

 