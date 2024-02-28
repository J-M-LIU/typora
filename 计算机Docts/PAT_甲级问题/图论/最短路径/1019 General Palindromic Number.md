## 1019 General Palindromic Number (20 分)

A number that will be the same when it is written forwards or backwards is known as a **Palindromic Number**. For example, 1234321 is a palindromic number. All single digit numbers are palindromic numbers.

Although palindromic numbers are most often considered in the decimal system, the concept of palindromicity can be applied to the natural numbers in any numeral system. Consider a number *N*>0 in base *b*≥2, where it is written in standard notation with *k*+1 digits *a**i* as ∑*i*=0*k*(*a**i**b**i*). Here, as usual, 0≤*a**i*<*b* for all *i* and *a**k* is non-zero. Then *N* is palindromic if and only if *a**i*=*a**k*−*i* for all *i*. Zero is written 0 in any base and is also palindromic by definition.

Given any positive decimal integer *N* and a base *b*, you are supposed to tell if *N* is a palindromic number in base *b*.

### Input Specification:

Each input file contains one test case. Each case consists of two positive numbers *N* and *b*, where 0<*N*≤109 is the decimal number and 2≤*b*≤109 is the base. The numbers are separated by a space.

### Output Specification:

For each test case, first print in one line `Yes` if *N* is a palindromic number in base *b*, or `No` if not. Then in the next line, print *N* as the number in base *b* in the form "*a**k* *a**k*−1 ... *a*0". Notice that there must be no extra space at the end of output.

### Sample Input 1:

```in
27 2
```

### Sample Output 1:

```out
Yes
1 1 0 1 1
```

### Sample Input 2:

```in
121 5
```

### Sample Output 2:

```out
No
4 4 1
```

```C++
#include<cstdio>
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

int main(){
    int N, radix;
    cin >> N >> radix;
    vector<int> arr;
    int temp = N, mod;
    while(temp!=0){
        mod = temp % radix;
        temp = temp / radix;
        arr.push_back(mod);
    }
    auto it = arr.begin();
    auto r_it = arr.rbegin();
    bool result = true;
    for (it, r_it; it != arr.end(), r_it != arr.rend();it++,r_it++){
        if(*it!=*r_it){
            result = false;
            break;
        }
    }
    string str = (result == true) ? "Yes" : "No";
    cout << str << endl;
    for (auto i = arr.rbegin(); i != arr.rend()-1;i++){
        cout << *i << " ";
    }
    cout << arr[0];
}
```

