## 1017 Queueing at Bank (25 分)

Suppose a bank has *K* windows open for service. There is a yellow line in front of the windows which devides the waiting area into two parts. All the customers have to wait in line behind the yellow line, until it is his/her turn to be served and there is a window available. It is assumed that no window can be occupied by a single customer for more than 1 hour.

Now given the arriving time *T* and the processing time *P* of each customer, you are supposed to tell the average waiting time of all the customers.

### Input Specification:

Each input file contains one test case. For each case, the first line contains 2 numbers: *N* (≤104) - the total number of customers, and *K* (≤100) - the number of windows. Then *N* lines follow, each contains 2 times: `HH:MM:SS` - the arriving time, and *P* - the processing time in minutes of a customer. Here `HH` is in the range [00, 23], `MM` and `SS` are both in [00, 59]. It is assumed that no two customers arrives at the same time.

Notice that the bank opens from 08:00 to 17:00. Anyone arrives early will have to wait in line till 08:00, and anyone comes too late (at or after 17:00:01) will not be served nor counted into the average.

### Output Specification:

For each test case, print in one line the average waiting time of all the customers, in minutes and accurate up to 1 decimal place.

### Sample Input:

```in
7 3
07:55:00 16
17:00:01 2
07:59:59 15
08:01:00 60
08:00:00 30
08:00:02 2
08:03:00 10
```

### Sample Output:

```out
8.2
```

```C++
#include<cstdio>
#include<iostream>
#include<queue>
#include<algorithm>
using namespace std;
struct Customer{
    int time;int process;
};
bool cmp(Customer c1,Customer c2){
    return c1.time < c2.time;
}
vector<Customer> customers;
vector<int> windows;
vector<int> waiting;
int main(){
    int N, K, process, hour, minute, second; //K个窗口，N个顾客
    cin >> N >> K;
    windows = vector<int>(K,8*3600);
    waiting = vector<int>(N, -1);
    for (int i = 0; i < N;i++){
        scanf("%02d:%02d:%02d %d", &hour, &minute, &second,&process);
        if(process>60)
            process = 60;
        int time = hour * 3600 + minute * 60 + second;
        if(time>17*3600)
            continue;
        Customer c;
        c.time = time;c.process=process;
        customers.push_back(c);
    }
    sort(customers.begin(), customers.end(), cmp);
    for (int i = 0; i < customers.size();i++){
        int temp = 0, min = windows[0];
        for (int j = 1; j < K;j++){
            if(windows[j]<min){
                temp = j;
                min = windows[j];
            }
        }
        // if (windows[temp] > 17 * 3600){
        //     //这里注意：不可将i之后的所有customer的waiting时间设置为17：00-到达时间，有可能后面的顾客存在晚于17点到达
        //     waiting[i] = 17 * 3600 - customers[i].time;
        // }
        // else{
            int wait = windows[temp] - customers[i].time;
            waiting[i] = (wait > 0) ? wait : 0;
            windows[temp] = (wait > 0) ? windows[temp] : customers[i].time;
            windows[temp] += customers[i].process * 60;
        // }
    }
    float sum = 0;  int index = 0;
    for (int i = 0; i < customers.size(); i++)
    {
        sum += waiting[i];index++;
    }
    sum = (index == 0) ? 0 : sum / 60.0 / index;
    printf("%.1f", sum);
}
```

