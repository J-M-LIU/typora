## 1009 Product of Polynomials (25 分)

This time, you are supposed to find *A*×*B* where *A* and *B* are two polynomials.

### Input Specification:

Each input file contains one test case. Each case occupies 2 lines, and each line contains the information of a polynomial:

*K* *N*1 *aN*1 *N*2 *aN*2 ... NK *a**N**K*

where *K* is the number of nonzero terms in the polynomial, *Ni* and *a**N**i* (*i*=1,2,⋯,*K*) are the exponents and coefficients, respectively. It is given that 1≤*K*≤10, 0≤*N**K*<⋯<*N*2<*N*1≤1000.

### Output Specification:

For each test case you should output the product of *A* and *B* in one line, with the same format as the input. Notice that there must be **NO** extra space at the end of each line. Please be accurate up to 1 decimal place.

### Sample Input:

```in
2 1 2.4 0 3.2
2 2 1.5 1 0.5
```

### Sample Output:

```out
3 3 3.6 2 6.0 1 1.6
```

```
#include<cstdio>
#include<iostream>
#include<map>
#include<iterator>
using namespace std;
 
int main(){
    map<int, float> arr1, arr2, arr;
    int K1, K2;
    int N;
    float a;
    cin >> K1;
    for (int i = 0; i < K1;i++){
        cin >> N >> a;
        arr1[N] = a;
    }
    cin >> K2;
    for (int i = 0; i < K2;i++){
        cin >> N >> a;
        arr2[N] = a;
    }
    for (map<int, float>::iterator iter1 = arr1.begin(); iter1 != arr1.end();iter1++){
        for(map<int, float>::iterator iter2 = arr2.begin(); iter2 != arr2.end();iter2++){
            N = iter1->first + iter2->first;
            a = iter1->second * iter2->second;
            arr[N] += a;
        }
    }
    for (auto iter = arr.begin(); iter != arr.end();iter++){
        if(iter->second==0)
            arr.erase(iter);
    }
        cout << arr.size();
    for(map<int, float>::reverse_iterator iter = arr.rbegin(); iter != arr.rend();iter++){
        printf(" %d %.1f", iter->first, iter->second);
    }
}
//注意a可能有负数的情况，消除a=0需要在最后进行
```

