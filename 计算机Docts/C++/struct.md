## struct结构体

### 结构体赋值

```C++
struct student{
	int number;
	string name;
	int age;
};

int main(){
	student s={1830,"刘佳敏",21};
	//或者
	student s;
	s.number=1830;
	s.name="刘佳敏";
	s.age=21;
}
```

### 结构体指针——初始化

```C++
struct BitNode{
	int data;
	BitNode* left;
	BitNode* right;
};
int main(){
	//结构体指针一定要new初始化，否则报错
	BitNode* tree=new BitNode();
	/*-----------*/
}
```

### 结构体构造函数

```C++
struct student{
	int number;
	string name;
	int age;
	//有参数的构造函数，记得再写上无参数的构造函数,规范化
	student():number(),name(),age(){}
	student(int n,string n,int a):number(n),name(n),age(a){}
};
创建对象
    Student student(18301134,"刘佳敏",20);
或
    Student student=Student(18301134,"刘佳敏",20);
如果是结构体指针：
    BitNode* Tree=new BitNode(); 一定要用new 初始化分配地址
```

