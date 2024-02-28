## 1014 Waiting in Line (30 分)

Suppose a bank has *N* windows open for service. There is a yellow line in front of the windows which devides the waiting area into two parts. The rules for the customers to wait in line are:

- The space inside the yellow line in front of each window is enough to contain a line with *M* customers. Hence when all the *N* lines are full, all the customers after (and including) the (*NM*+1)st one will have to wait in a line behind the yellow line.
- Each customer will choose the shortest line to wait in when crossing the yellow line. If there are two or more lines with the same length, the customer will always choose the window with the smallest number.
- *C**u**s**t**o**m**e**r**i* will take *T**i* minutes to have his/her transaction processed.
- The first *N* customers are assumed to be served at 8:00am.

Now given the processing time of each customer, you are supposed to tell the exact time at which a customer has his/her business done.

For example, suppose that a bank has 2 windows and each window may have 2 customers waiting inside the yellow line. There are 5 customers waiting with transactions taking 1, 2, 6, 4 and 3 minutes, respectively. At 08:00 in the morning, *c**u**s**t**o**m**er*1 is served at *w**in**d**o**w*1 while *c**u**s**t**o**m**er*2 is served at *w**in**d**o**w*2. *C**u**s**t**o**m**er*3 will wait in front of *w**in**d**o**w*1 and *c**u**s**t**o**m**er*4 will wait in front of *w**in**d**o**w*2. *C**u**s**t**o**m**er*5 will wait behind the yellow line.

At 08:01, *c**u**s**t**o**m**e**r*1 is done and *c**u**s**t**o**m**er*5 enters the line in front of *w**in**d**o**w*1 since that line seems shorter now. *C**u**s**t**o**m**e**r*2 will leave at 08:02, *c**u**s**t**o**m**er*4 at 08:06, *c**u**s**t**o**m**e**r*3 at 08:07, and finally *c**u**s**t**o**m**er*5 at 08:10.

### Input Specification:

Each input file contains one test case. Each case starts with a line containing 4 positive integers: *N* (≤20, number of windows), *M* (≤10, the maximum capacity of each line inside the yellow line), *K* (≤1000, number of customers), and *Q* (≤1000, number of customer queries).

The next line contains *K* positive integers, which are the processing time of the *K* customers.

The last line contains *Q* positive integers, which represent the customers who are asking about the time they can have their transactions done. The customers are numbered from 1 to *K*.

### Output Specification:

For each of the *Q* customers, print in one line the time at which his/her transaction is finished, in the format `HH:MM` where `HH` is in [08, 17] and `MM` is in [00, 59]. Note that since the bank is closed everyday after 17:00, for those customers who cannot be served before 17:00, you must output `Sorry` instead.

### Sample Input:

```in
2 2 7 5
1 2 6 4 3 534 2
3 4 5 6 7
```

### Sample Output:

```out
08:07
08:06
08:10
17:00
Sorry
```

```C++
#include<cstdio>
#include<iostream>
#include<queue>
#include<vector>
using namespace std;
struct Window{
    int poptime;//队首的人的出队时间
    int endtime;//队尾的人的结束时间
    queue<int> wait;
};
int N, M, K, Q; //N个窗口，M最大容量，K个顾客，Q个查询
vector<Window> windows; 
vector<int> process;//每个顾客的处理时间
vector<int> finish; //每个顾客的结束处理时间
int main(){
    cin >> N >> M >> K >> Q;
    Window w;
    w.endtime = 0;w.poptime=0;
    windows = vector<Window>(N + 1, w);
    finish = vector<int>(K + 1, 0);
    process.push_back(0);
    for (int i = 1; i <=K;i++){    //获取每个人处理时间
        int n;
        cin >> n;
        process.push_back(n);
    }
    //先排N*M个人
    int index = 1;
    for (int i = 1; i <= M;i++){  //一行一行地排列
        for (int j = 1; j <= N;j++){ //每一行有N个窗口 第j个窗口
            if(index<=K){
                windows[j].wait.push(index);
                if(windows[j].endtime>=540){
                    windows[j].endtime += process[index];
                    finish[index] = -1;
                }
                else{
                    windows[j].endtime += process[index];
                    finish[index] = windows[j].endtime;
                }
                if (i == 1){
                    windows[j].poptime = process[index];
                }
                index++;
            }
            
        }
    }
    //排剩下的K-NM个人
    while(index<=K){
        int temp = 1, min=windows[1].poptime;
        //找到poptime最小的一个窗口
        for(int i = 2; i <= N;i++){
            if(windows[i].poptime<min){
                temp = i;
                min = windows[i].poptime;
            }
        }
        windows[temp].wait.push(index);   //记得不要忘了push(index)
        windows[temp].wait.pop();    
        windows[temp].poptime += process[windows[temp].wait.front()];
        if(windows[temp].endtime>=540){
            finish[index] = -1;
            windows[temp].endtime += process[index];
        }
        else{
            windows[temp].endtime += process[index];
            finish[index] = windows[temp].endtime;
        }
        index++;
    }
    int q, hour, minute;
    for (int i = 0; i < Q;i++){
        cin >> q;
        if(finish[q]==-1)
            cout << "Sorry" << endl;
        else{
            hour = finish[q] / 60 + 8;
            minute = finish[q] % 60;
            printf("%02d:%02d\n", hour, minute);
        }
    }
}
```

