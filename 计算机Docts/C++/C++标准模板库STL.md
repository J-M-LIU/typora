# Standard Template Library, STL 

[TOC]

## 序列容器

vector、list、deque、queue、stack、优先级队列

| **表 达 式**        | **返 回 类 型** | **说 明**                        |
| ------------------- | --------------- | -------------------------------- |
| **a.insert(p,t)**   | **迭代器**      | **将t插入到p的前面**             |
| **a.insert(p,n,t)** | **void**        | **将n个t插入到p的前面**          |
| **a.insert(p,i,j)** | **void**        | **将区间[i,j)的元素插入到p前面** |
| **a.erase(p)**      | **迭代器**      | **删除p所指向的元素**            |
| **a.erase(p,q)**    | **迭代器**      | **删除区间[p,q)中的元素**        |
| **a.clear()**       | **void**        | **清空容器**                     |



### Vector-向量-变长数组

**vector强调的是快速查找，而list强调的是快速插入和删除**

#### 使用

```
#include<vector>
#include<cstdio>
#include<algorithm>   //使用for_each()
#include<iterator>    //使用迭代器
using namespace std;
```

#### 定义

```c++
vector<typename> name
```

#### 指定值初始化

```C++
vector<int> v (10,0);//初始化为包含10个0的vector
```

#### 拷贝初始化

```C++
vector<int> v2(v1);//v1和v2必须类型相同
vector<int> v2=v1;
```

#### 指定范围初始化

```
vector<int> v2(v1.begin()+5,v1.end()-5);
```

#### vector实现邻接表

```c++
const int SIZE 10000;
//N表示 图结构有N个顶点，每个Adj[i]就是一个变长数组，存储空间只与图的边数有关,M表示有M条边
	int N, M;
	scanf("%d %d", &N,&M);
	//必须用常量分配
	vector<int> Adj[SIZE];
	int start, end;
	//建表
	for (int i = 0; i < M; i++) {
		scanf("%d %d", &start, &end);
		Adj[start].push_back(end);
		Adj[end].push_back(start); //有向图则不考虑此行
	}
	//遍历
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < Adj[i].size(); j++) {
			printf("%d", Adj[i][j]);
		}
		printf("\n");
	}
```

#### 遍历

两种方法

（1）for循环迭代器输出

```C++
 for (vector<int>::const_iterator iter = valList.cbegin(); iter != valList.cend(); iter++)
    {
        cout << (*iter) << endl;
    }

vectot<int>::iterator iter=valList.begin();
while(iter!=valList.end()){
    cout<<*iter<<endl;
}
```

（2）for_each加输出函数

```
void output(int n){
	cout<<n<<endl;
}
for_each(vec.begin(),vec.end(),output);
```

#### 插入

只有在vector，string中，可以使用 迭代器 arr.begin()+number的用法

```
vec.insert(vec.begin()+3,-1); //将-1插入vec[3]的位置
```



### list

#### 使用

```
#include<list>
using namespace std;
```

```C++
scanf("%d", &x);
	while (x != -9999) {
		//尾插法
		arr.push_back(x);
		scanf("%d", &x);
	}
	//遍历
	for_each(arr.begin(), arr.end(), output);
	
	//迭代器
	list<int>::iterator it = arr.begin();
	//找到需要插入的位置
	int key = 20;
	while (*it != key && it!=arr.end())
		it++;
	//插入元素 50：这是在key=20处前面插入
	arr.insert(it, 50);

	//删除iterator处的元素 此时保留的仍是key=20的迭代器地址
	arr.erase(it);

	//给list排序，复杂度NlogN
	arr.sort();

	//arr.merge(list2),将list2合并到arr中，list2将为空
	return 0;
```

### priority_queue

**和`queue`不同的就在于我们可以自定义其中数据的优先级, 让优先级高的排在队列前面,优先出队。**

**优先队列具有队列的所有特性，包括队列的基本操作，只是在这基础上添加了内部的一个排序，它本质是一个堆实现的。**

**定义：priority_queue<Type, Container, Functional>**
Type 就是数据类型，Container 就是容器类型（Container必须是用数组实现的容器，比如vector,deque等等，但不能用 list。STL里面默认用的是vector），Functional 就是比较的方式。

```C++
#include<queue>
//升序队列，小顶堆
priority_queue<int,vector<int>,greater<int>>
//降序队列，大顶堆
priority_queue<int,vector<int>,less<int>>
```



## algorithm头文件下的常用函数

#### max(), min(), abs()

返回x,y,z的最大值

可用max(x,max(y,z))

#### swap(x,y)

用于交换x和y的值

#### reverse(a1,a2)

a1和a2代表首元素地址和尾元素地址的下一个地址 [a1,a2)

int a[5]={1,2,3,4,5}

reverse(a,a+4)

<u>4,2,3,1,5</u>

#### next_permutation()

给出一个序列在全排列中的下一个序列

```c++
int a[3]={1,2,3};
do{
	printf("%d %d %d",a[0],a[1],a[2])
}while(next_permutation(a,a+3))
```

result: 123 132 213 231 312 321

#### fill(addr,addr+k,value)

将数组或容器中的某一段区间赋为某个相同的值

#### sort()

- 不推荐使用C语言下的qsort()，因为有很多指针操作

- C++的sort()规避了快速排序极端情况下退化到O(n^2^)的可能

##### 使用条件

#include<Algorithm> 

using namespace std;

##### 参数

sort(首元素地址，尾元素地址的下一个地址，比较函数(非必填));

<u>*[首，尾)*</u>  美国人习惯 左闭右开

```C++
//数组arr[] 
sort(arr,arr+length,cmp);
//容器 list<int> l
sort(l.begin(),l.end(),cmp);

```



## 反向迭代器

```C++
//map<int,int>::reverse_iterator iter=m.rbegin();
for(map<int,int>::reverse_iterator iter=m.rbegin();iter!=m.rend();iter++){
	...
}
```

 

### max_element() / min_element() 

**这两个函数返回最大值或最小值的迭代器，所以要加***

#### 数组写法

```C++
int arr[10]={1,2,3,4,5,6,7,8,9,0};
int max=*max_element(arr,arr+10);
int min=*min_element(arr,arr+10);
```

#### STL容器写法

```C++
vector<int> v;
int max=*max_element(v.begin(),v.end());
int min=*min_element(v.begin(),v.end());
或者如果要获取迭代器
auto it=max_element(v.begin(),v.end());
```



## 优先队列 priority_queue

#### 定义：priority_queue<Type, Container, Functional>

Type 就是数据类型，Container 就是容器类型（Container必须是用数组实现的容器，比如vector,deque等等，但不能用 list。STL里面默认用的是vector），Functional 就是比较的方式，当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，默认是大顶堆。

```C++
//升序队列
priority_queue <int,vector<int>,greater<int> > q;
//降序队列
priority_queue <int,vector<int>,less<int> >q;
//greater和less是std实现的两个仿函数（就是使一个类的使用看上去像一个函数。其实现就是类中实现一个operator()，这个类就有了类似函数的行为，就是一个仿函数类了）

```

#### 对于自定义类型

```C++
#include <iostream>
#include <queue>
using namespace std;

//方法1
struct tmp1 //运算符重载<
{
    int x;
    tmp1(int a) {x = a;}
    bool operator<(const tmp1& a) const
    {
        return x < a.x; //大顶堆
    }
};

//方法2
struct tmp2 //重写仿函数
{
    bool operator() (tmp1 a, tmp1 b) 
    {
        return a.x < b.x; //大顶堆
    }
};

int main() 
{
    tmp1 a(1);
    tmp1 b(2);
    tmp1 c(3);
    priority_queue<tmp1> d;
    d.push(b);
    d.push(c);
    d.push(a);
    while (!d.empty()) 
    {
        cout << d.top().x << '\n';
        d.pop();
    }
    cout << endl;

    priority_queue<tmp1, vector<tmp1>, tmp2> f;
    f.push(c);
    f.push(b);
    f.push(a);
    while (!f.empty()) 
    {
        cout << f.top().x << '\n';
        f.pop();
    }
}
```

## upper_bound&lower_bound

**upper_bound():**  返回的是被查序列中第一个大于查找值的指针；

**lower_bound()**：返回的是被查序列中第一个大于等于查找值的指针；

```
int arr[100];
int pos=lower_bound(arr,arr+100,n)
vector<int> arr;
int number=*uppper_bound(arr.begin(),arr.end(),n);
```

#### lower_bound

1.如果m在区间中没有出现过，那么返回第一个比m大的数的下标。
2.如果m比所有区间内的数都大，那么返回r。这个时候会越界，小心。
3.如果区间内有多个相同的m，返回第一个m的下标。

#### upper_bound

1.如果m在区间中没有出现过，那么返回第一个比m大的数的下标。
2.如果m比所有区间内的数都大，那么返回r。这个时候会越界，小心。
3.如果区间内有多个相同的m，返回最后一个m的下标+1。



### count()

##### map.count(type key)  

查看是否包含key,由于map不包含重复的key，因此m.count(key)取值为0，或者1，表示是否包含

##### map.find(type key)

返回迭代器，判断是否存在

set的这两个成员函数与map一致

##### set.count(type key)

##### set.find(type key)  返回迭代器

#### 泛型函数 

##### count(begin,last,value) 

线性的方法，返回容器内value的个数

##### find(begin,last,value)

返回第一次找到value的指针



## 注意auto和iterator的区别

auto具有迭代器的作用，但不是一个指针；但通过it=m.begin()这种方法获取的迭代器，是相当于一个指针的。如下是map

```C++
map<int,int> m;
for(auto it:m){
	cout<<it.first<<it.second;
}
for(auto it=m.begin();it!=m.end();it++){  //或者map<int,int>::iterator it=m.begin();
	cout<<it->first<<it->second;
}
```

vector中

```C++
vector<int> v;
for(auto it:v){
	cout<<it<<endl;
}
for(auto it=v.begin();it!=v.end();it++){
	cout<<*it;
}
```



## erase用法

#### map

##### map.erase(iterator it)   通多迭代器消除

##### map.erase(type key)     根据键值消除

