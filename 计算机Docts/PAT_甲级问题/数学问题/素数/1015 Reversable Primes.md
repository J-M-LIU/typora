## 1015 Reversible Primes (20 分)

A **reversible prime** in any number system is a prime whose "reverse" in that number system is also a prime. For example in the decimal system 73 is a reversible prime because its reverse 37 is also a prime.

Now given any two positive integers *N* (<105) and *D* (1<*D*≤10), you are supposed to tell if *N* is a reversible prime with radix *D*.

### Input Specification:

The input file consists of several test cases. Each case occupies a line which contains two integers *N* and *D*. The input is finished by a negative *N*.

### Output Specification:

For each test case, print in one line `Yes` if *N* is a reversible prime with radix *D*, or `No` if not.

### Sample Input:

```in
73 10
23 2
23 10
-2结尾无空行
```

### Sample Output:

```out
Yes
Yes
No
```



```C++
#include<cstdio>
#include<iostream>
#include<cmath>
#include<string>
#include<algorithm>
using namespace std;
int convert(int num,int radix){
    string str;
    int temp = num, mod;
    while(temp!=0){
        mod = temp % radix;
        temp = temp / radix;
        str = to_string(mod).append(str);
    }
    reverse(str.begin(), str.end());
    //转化为10进制
    int sum = 0, index = 0;
    for (auto iter = str.rbegin(); iter != str.rend();iter++){
        sum += (*iter-'0') * pow(radix, index++);
    }
    return sum;
}
bool isPrime(int num){
    if(num==0||num==1)
        return false;
    int bound = sqrt(num);
    int i = 2;
    while(i<=bound){
        if(num%i==0)
            return false;
        i++;
    }
    return true;
}
int main(){
    int N, D;
    while(cin>>N){
        if(N>0){
            cin >> D;
            if(isPrime(N)){
                cout << (isPrime(convert(N, D)) ? "Yes" :"No")<<endl;
            }
            else
                cout << "No" << endl;
        }else
            break;
    }
    return 0;
}
```

