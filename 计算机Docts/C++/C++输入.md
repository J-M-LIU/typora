### 字符数组char[]

#### getchar()获取字符

```C++
char str[1024];
int i=0;
while((str[i]=getchar())!='\n')
    i++;
str[i]='\0'
```

#### gets()读入char[]

```c++
char str[1024];
gets(str);
```

**不用考虑换行符，结尾为字符串结束标志符'\0'**

#### cin.getline(char[], int length)

```
char str[100];
cin.getline(str,100);
```

getline()会自动消除换行符

#### cin.get(char[], int length)

```C++
char str[100];
cin.get(str,100);
```

getline()会自动丢弃换行符，而get()将保留，使用cin.get()获取多行数据时，中间再使用get()消除换行符

```C++
char input[1024],str[30];
cin.get(input, 1024);
cin.get();//如果这一句被注释掉，那么在输入的时候，上一句输入完回车，下一句局不会执行。
cin.get(str, 30);
```

### string类

```C++
//getline():
string str;
getline(cin,str);
```

**getline()第三个参数表示间隔符，默认为换行符'\n'。读入不需要考虑最后的换行符。**

***注意：cin>>str和getline(cin,str)不要混用，如果前面的输入是cin>>ss, 那么此处getline(cin,str)中str的值是空的，因为会读取上一行的结束符。***如果混用需要通过getchar() 或cin.get()消除换行符



### 一行中按空格分开 获取多次输入

```C++
如 “小说 科幻 玄幻 悬疑 中篇 热销” 
while(cin>>keyword){
	//do something
	char c=getchar();  //输入以换行符结束
	if(c=='\n')
		break;
}
```

### cin\scanf与getline混用的情况

getline会自动抛弃换行符

所以cin或scanf使用后，如果有换行，加上 **cin.get()**或 **getchar()**来消除换行符 或者**scanf("%d\n", &n);**

注意cin与scanf混用不需要手动消除换行符

