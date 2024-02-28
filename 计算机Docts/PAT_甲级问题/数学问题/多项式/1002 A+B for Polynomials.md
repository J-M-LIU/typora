## 1002 A+B for Polynomials (25 分)

This time, you are supposed to find *A*+*B* where *A* and *B* are two polynomials.

### Input Specification:

Each input file contains one test case. Each case occupies 2 lines, and each line contains the information of a polynomial:

*K* *N*1 *aN*1 *N*2 *aN*2 ... NK *a**N**K*

where *K* is the number of nonzero terms in the polynomial, *N**i* and *a**N**i* (*i*=1,2,⋯,*K*) are the exponents and coefficients, respectively. It is given that 1≤*K*≤10，0≤*NK*<⋯<*N*2<*N*1≤1000.

### Output Specification:

For each test case you should output the sum of *A* and *B* in one line, with the same format as the input. Notice that there must be NO extra space at the end of each line. Please be accurate to 1 decimal place.

### Sample Input:

```in
2 1 2.4 0 3.2
2 2 1.5 1 0.5
```

### Sample Output:

```out
3 2 1.5 1 2.9 0 3.2
```

```C++
#include<cstdio>
#include<iostream>
#include<algorithm>
#include<map>
using namespace std;
int main(){
    map<float, float> m;
    int K1, K2;
    float N;
    float a;
    cin >> K1;
    for (int i = 0; i < K1;i++){
        cin >> N >> a;
        m[N] += a;
    }
    cin >> K2;
    for (int i = 0; i < K2;i++){
        cin >> N >> a;
        m[N] += a;
        //系数a为0的情况要删去
        if(m[N]==0)
            m.erase(N);
    }
    cout << m.size();
    for (map<float, float>::reverse_iterator iter = m.rbegin(); iter != m.rend();iter++){
        printf(" %.0f %.1f", iter->first, iter->second);
    }
}
```

