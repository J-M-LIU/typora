## 1098 Insertion or Heap Sort (25 分)

According to Wikipedia:

**Insertion sort** iterates, consuming one input element each repetition, and growing a sorted output list. Each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list, and inserts it there. It repeats until no input elements remain.

**Heap sort** divides its input into a sorted and an unsorted region, and it iteratively shrinks the unsorted region by extracting the largest element and moving that to the sorted region. it involves the use of a heap data structure rather than a linear-time search to find the maximum.

Now given the initial sequence of integers, together with a sequence which is a result of several iterations of some sorting method, can you tell which sorting method we are using?

### Input Specification:

Each input file contains one test case. For each case, the first line gives a positive integer *N* (≤100). Then in the next line, *N* integers are given as the initial sequence. The last line contains the partially sorted sequence of the *N* numbers. It is assumed that the target sequence is always ascending. All the numbers in a line are separated by a space.

### Output Specification:

For each test case, print in the first line either "Insertion Sort" or "Heap Sort" to indicate the method used to obtain the partial result. Then run this method for one more iteration and output in the second line the resulting sequence. It is guaranteed that the answer is unique for each test case. All the numbers in a line must be separated by a space, and there must be no extra space at the end of the line.

### Sample Input 1:

```in
10
3 1 2 8 7 5 9 4 6 0
1 2 3 7 8 5 9 4 6 0结尾无空行
```

### Sample Output 1:

```out
Insertion Sort
1 2 3 5 7 8 9 4 6 0结尾无空行
```

### Sample Input 2:

```
10
3 1 2 8 7 5 9 4 6 0
6 4 5 1 0 3 2 7 8 9
```

### Sample Output 2:

```
Heap Sort
5 4 3 1 0 2 6 7 8 9
```

1. **判断是否是插入排序：a为原始序列，b为排序后序列，：b数组前⾯的顺序是从⼩到⼤的，后⾯的顺序不⼀定，但是⼀定和原序列的 后⾯的顺序相同～所以只要遍历⼀下前⾯⼏位，遇到不是从⼩到⼤的时候，开始看b和a是不是对应位 置的值相等，相等就说明是插⼊排序，否则就是堆排序**
2. 这里注意堆排序的算法

```
#include<cstdio>
#include<iostream>
#include<algorithm>
using namespace std;
int n;
int arr[110],cpy[110],sorted[110];
void downadjust(int low,int high){
    int i=low,j=2*low;
    while(j<=high){
        if(j+1<=high&&sorted[j+1]>sorted[j])
            j=j+1;
        if (sorted[j] > sorted[i]) {
            swap(sorted[i], sorted[j]);
            i=j;
            j=2*i;
        }
        else break;  //如果孩子结点的值都比父结点小，则说明不需继续调整
    }
}
int main(){
    cin>>n;
    for(int i=1;i<=n;i++)cin>>arr[i];
    for(int i=1;i<=n;i++)cin>>sorted[i];
    int index=2;
    while(index<=n){
        if(sorted[index]<sorted[index-1]) break;
        index++;
    }
    bool flag=1;
    for(int i=index;i<=n;i++){
        if(sorted[i]!=arr[i]){flag=0;
            printf("Heap Sort\n");
            sort(arr+1,arr+n+1);
            index=n;
            while(index>=1){
                if(arr[index]!=sorted[index]) break;
                index--;
            }
            swap(sorted[1],sorted[index]);
            downadjust(1,index-1);
            break;
        }
    }
    if(flag){
        printf("Insertion Sort\n");
        sort(sorted+1,sorted+index+1);
    }
    for(int i=1;i<=n;i++){
        printf("%d",sorted[i]);
        if(i<n) printf(" ");
    }   
}
```

