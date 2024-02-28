## 1010 Radix (25 分)

Given a pair of positive integers, for example, 6 and 110, can this equation 6 = 110 be true? The answer is `yes`, if 6 is a decimal number and 110 is a binary number.

Now for any pair of positive integers *N*1 and *N*2, your task is to find the radix of one number while that of the other is given.

### Input Specification:

Each input file contains one test case. Each case occupies a line which contains 4 positive integers:

```
N1 N2 tag radix
```

Here `N1` and `N2` each has no more than 10 digits. A digit is less than its radix and is chosen from the set { 0-9, `a`-`z` } where 0-9 represent the decimal numbers 0-9, and `a`-`z` represent the decimal numbers 10-35. The last number `radix` is the radix of `N1` if `tag` is 1, or of `N2` if `tag` is 2.

### Output Specification:

For each test case, print in one line the radix of the other number so that the equation `N1` = `N2` is true. If the equation is impossible, print `Impossible`. If the solution is not unique, output the smallest possible radix.

### Sample Input 1:

```in
6 110 1 10
```

### Sample Output 1:

```out
2
```

### Sample Input 2:

```in
1 ab 1 2
```

### Sample Output 2:

```out
Impossible
```

```C++
//二分查找
#include<cstdio>
#include<iostream>
#include<cmath>
#include<cctype>
#include<algorithm>
using namespace std;
//将radix进制的数转化为10进制
long long convert(string s,long long radix){
    long long sum = 0;
    int index = 0, num;
    for (auto iter = s.rbegin(); iter != s.rend();iter++){
        num = isdigit(*iter) ? (*iter - '0') : (*iter - 'a' + 10);
        sum += num * pow(radix, index++);
    }
        return sum;
}
//如果转换后的值为负值，说明超出long long的范围，radix过大
/*进制的最大值不是36，而是target的值。如target=15，若s="10",radix=15即可, 
若s="11",radix=14，s越大，进制radix就会越小于target*/
long long binarySort(long long target,string s){
    auto iter = max_element(s.begin(), s.end());
    long long left = (isdigit(*iter) ? (*iter - '0') : (*iter - 'a' + 10) )+ 1;
    long long right = max(target, left);
    long long mid;
    while(left<=right){
        mid = (left + right) / 2;
        long long m = convert(s, mid);
        if(m==target)
            return mid;
        else if(target<m||m<0) //转换得到的数字>target或者出现负值，说明当前进制太大
            right = mid - 1;
        else 
            left = mid + 1;
    }
    return -1;
}
int main(){
    long long result_radix, tag, radix;
    string N1, N2;
    cin >> N1 >> N2 >> tag >> radix;
    result_radix = tag == 1 ? binarySort(convert(N1, radix), N2) : 
    binarySort(convert(N2, radix), N1);
    if(result_radix==-1)
        cout << ("Impossible");
    else
        cout << result_radix;
    return 0;
}

```

