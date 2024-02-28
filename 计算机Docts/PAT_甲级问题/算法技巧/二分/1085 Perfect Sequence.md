## 1085 Perfect Sequence (25 分)

Given a sequence of positive integers and another positive integer *p*. The sequence is said to be a **perfect sequence** if *M*≤*m*×*p* where *M* and *m* are the maximum and minimum numbers in the sequence, respectively.

Now given a sequence and a parameter *p*, you are supposed to find from the sequence as many numbers as possible to form a perfect subsequence.

### Input Specification:

Each input file contains one test case. For each case, the first line contains two positive integers *N* and *p*, where *N* (≤105) is the number of integers in the sequence, and *p* (≤109) is the parameter. In the second line there are *N* positive integers, each is no greater than 109.

### Output Specification:

For each test case, print in one line the maximum number of integers that can be chosen to form a perfect subsequence.

### Sample Input:

```in
10 8
2 3 20 4 5 1 6 7 8 9结尾无空行
```

### Sample Output:

```out
8
```

```C++
#include<cstdio>
#include<iostream>
#include<algorithm>
using namespace std;
int n; long long p;
long long arr[100010];
//two pointers思想:二分思想
int main(){
    cin>>n>>p;
    for(int i=0;i<n;i++)cin>>arr[i];
    sort(arr,arr+n);
    int temp=0;
    for(int i=0;i<n;i++){
        for(int j=i+temp;j<n;j++){
            if(arr[j]<=arr[i]*p) temp++;
            else break;
        }
    }
    cout<<temp;
}
```

**另一种做法 巧用upper_bound()**

```C++
#include <cstdio>
#include <algorithm>
#include <cmath>
#define MAX_N 100000
using namespace std;
int main() {
    int N, P;
    int a[MAX_N];
    scanf("%d %d", &N, &P);
    for(int i = 0; i < N; i++){
        scanf("%d", &a[i]);
    }
    sort(a, a + N);
    int cnt = 0;
    for(int i = 0, j; i < N; i++){
        // 注意这里的类型转换，避免乘积溢出
        j = (int)(upper_bound(a + i + 1, a + N, (long long)a[i] * P) - a);
        cnt = max(cnt, j - i);
    }
    printf("%d", cnt);
    return 0;
}

```

