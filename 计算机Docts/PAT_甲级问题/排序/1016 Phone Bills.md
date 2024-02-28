## 1016 Phone Bills (25 分)

A long-distance telephone company charges its customers by the following rules:

Making a long-distance call costs a certain amount per minute, depending on the time of day when the call is made. When a customer starts connecting a long-distance call, the time will be recorded, and so will be the time when the customer hangs up the phone. Every calendar month, a bill is sent to the customer for each minute called (at a rate determined by the time of day). Your job is to prepare the bills for each month, given a set of phone call records.

### Input Specification:

Each input file contains one test case. Each case has two parts: the rate structure, and the phone call records.

The rate structure consists of a line with 24 non-negative integers denoting the toll (cents/minute) from 00:00 - 01:00, the toll from 01:00 - 02:00, and so on for each hour in the day.

The next line contains a positive number *N* (≤1000), followed by *N* lines of records. Each phone call record consists of the name of the customer (string of up to 20 characters without space), the time and date (`MM:dd:HH:mm`), and the word `on-line` or `off-line`.

For each test case, all dates will be within a single month. Each `on-line` record is paired with the chronologically next record for the same customer provided it is an `off-line` record. Any `on-line` records that are not paired with an `off-line` record are ignored, as are `off-line` records not paired with an `on-line` record. It is guaranteed that at least one call is well paired in the input. You may assume that no two records for the same customer have the same time. Times are recorded using a 24-hour clock.

### Output Specification:

For each test case, you must print a phone bill for each customer.

Bills must be printed in alphabetical order of customers' names. For each customer, first print in a line the name of the customer and the month of the bill in the format shown by the sample. Then for each time period of a call, print in one line the beginning and ending time and date (`dd:HH:mm`), the lasting time (in minute) and the charge of the call. The calls must be listed in chronological order. Finally, print the total charge for the month in the format shown by the sample.

### Sample Input:

```in
10 10 10 10 10 10 20 20 20 15 15 15 15 15 15 15 20 30 20 15 15 10 10 10
10
CYLL 01:01:06:01 on-line
CYLL 01:28:16:05 off-line
CYJJ 01:01:07:00 off-line
CYLL 01:01:08:03 off-line
CYJJ 01:01:05:59 on-line
aaa 01:01:01:03 on-line
aaa 01:02:00:01 on-line
CYLL 01:28:15:41 on-line
aaa 01:05:02:24 on-line
aaa 01:04:23:59 off-line
```

### Sample Output:

```out
CYJJ 01
01:05:59 01:07:00 61 $12.10
Total amount: $12.10
CYLL 01
01:06:01 01:08:03 122 $24.40
28:15:41 28:16:05 24 $3.85
Total amount: $28.25
aaa 01
02:00:01 04:23:59 4318 $638.80
Total amount: $638.80
```

```C++
#include<cstdio>
#include<iostream>
#include<map>
#include<string>
#include<algorithm>
#include<vector>
using namespace std;
struct record{
    int month;
    int day;
    int hour;
    int minute;
    string time;
    int state;
};
bool cmp(record r1,record r2){
    return r1.time < r2.time;
}
int rate[25];
map<string, vector<record>> calls;
//这里费用的计算，通过设置相同的时间起点，总的费用进行相减得到
float billFromZero(record r){
    //按照月初到现在的时间进行计算  09:08:53
    float sum = rate[24] * r.day * 60 + rate[r.hour] * r.minute;
    for (int i = 0; i < r.hour;i++){
        sum += rate[i] * 60;
    }
    return sum / 100.0;
}

int main(){
    //rate[24]存放一天24个时间段的费率和，用以进行跨天的计算
    for (int i = 0; i < 24;i++){
        cin >> rate[i];
        rate[24] += rate[i];
    }
    int N;
    cin >> N;
    string name, time,state;
    for (int i = 0; i < N;i++){
        cin >> name >> time >> state;
        record r;
        r.month = atoi(time.substr(0, 2).c_str());
        r.day = atoi(time.substr(3, 2).c_str());
        r.hour = atoi(time.substr(6, 2).c_str());
        r.minute = atoi(time.substr(9, 2).c_str());
        r.state = (state == "on-line") ? 1 : 0;
        r.time = time;
        calls[name].push_back(r);
    }
    for (auto iter = calls.begin(); iter != calls.end();iter++){
        sort(iter->second.begin(), iter->second.end(), cmp);
    }
    for (auto iter = calls.begin(); iter != calls.end(); iter++){
        float total = 0, fee = 0;
        auto it = iter->second.begin();
        for (it; it != iter->second.end()-1;it++){  //注意这里：it不可以加到end(),
                                                    //因为要判断it+1，否则访问不合法地址，结果错误
            if(it->state==1&&(it+1)->state==0){
                if(total==0){  //这里需要注意一个点：没有产生费用的用户不进行输出
                    cout << iter->first;
                    printf(" %02d\n", it->month);
                }
                fee = billFromZero(*(it + 1))-billFromZero(*it);
                total += fee;
                int mins = ((it + 1)->day - it->day) * 24 * 60 + ((it + 1)->hour - it->hour) * 60 +
                 ((it + 1)->minute - it->minute);
                printf("%02d:%02d:%02d %02d:%02d:%02d %d $%.2f\n", it->day, it->hour, it->minute,
                       (it + 1)->day, (it + 1)->hour, (it + 1)->minute,mins,fee);
            }
        }
        if(total!=0)
            printf("Total amount: $%.2f\n", total);
    }
}
```

