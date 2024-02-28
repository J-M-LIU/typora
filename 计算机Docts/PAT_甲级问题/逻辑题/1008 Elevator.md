## 1008 Elevator (20 分)

The highest building in our city has only one elevator. A request list is made up with *N* positive numbers. The numbers denote at which floors the elevator will stop, in specified order. It costs 6 seconds to move the elevator up one floor, and 4 seconds to move down one floor. The elevator will stay for 5 seconds at each stop.

For a given request list, you are to compute the total time spent to fulfill the requests on the list. The elevator is on the 0th floor at the beginning and does not have to return to the ground floor when the requests are fulfilled.

### Input Specification:

Each input file contains one test case. Each case contains a positive integer *N*, followed by *N* positive numbers. All the numbers in the input are less than 100.

### Output Specification:

For each test case, print the total time on a single line.

### Sample Input:

```in
3 2 3 1
```

### Sample Output:

```out
41
```

```C++
#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
const int stay = 5;
const int up = 6;
const int down = 4;
int main(){
    vector<int> request;
    int N, l;
    cin >> N;
    int time = 0;
    request.push_back(0);
    for (int i = 1; i <=N;i++){
        cin >> l;
        request.push_back(l);
        int diff = request[i] - request[i - 1];
        if(diff>0) //上楼
        {
            time += diff * up;
        }
        else if(diff<0){
            time += (abs(diff)) * down;
        }
    }
    time += N * stay;
    cout << time;
}
```

