## 1012 The Best Rank (25 分)

To evaluate the performance of our first year CS majored students, we consider their grades of three courses only: 

`C` - C Programming Language, `M` - Mathematics (Calculus or Linear Algrbra), and `E` - English. At the mean time, we encourage students by emphasizing on their best ranks -- that is, among the four ranks with respect to the three courses and the average grade, we print the best rank for each student.

For example, The grades of `C`, `M`, `E` and `A` - Average of 4 students are given as the following:

```
StudentID  C  M  E  A
310101     98 85 88 90
310102     70 95 88 84
310103     82 87 94 88
310104     91 91 91 91
```

Then the best ranks for all the students are No.1 since the 1st one has done the best in C Programming Language, while the 2nd one in Mathematics, the 3rd one in English, and the last one in average.

### Input Specification:

Each input file contains one test case. Each case starts with a line containing 2 numbers *N* and *M* (≤2000), which are the total number of students, and the number of students who would check their ranks, respectively. Then *N* lines follow, each contains a student ID which is a string of 6 digits, followed by the three integer grades (in the range of [0, 100]) of that student in the order of `C`, `M` and `E`. Then there are *M* lines, each containing a student ID.

### Output Specification:

For each of the *M* students, print in one line the best rank for him/her, and the symbol of the corresponding rank, separated by a space.

The priorities of the ranking methods are ordered as `A` > `C` > `M` > `E`. Hence if there are two or more ways for a student to obtain the same best rank, output the one with the highest priority.

If a student is not on the grading list, simply output `N/A`.

### Sample Input:

```in
5 6
310101 98 85 88
310102 70 95 88
310103 82 87 94
310104 91 91 91
310105 85 90 90
310101
310102
310103
310104
310105
999999
```

### Sample Output:

```out
1 C
1 M
1 E
1 A
3 A
N/A
```

```C++
#include<cstdio>
#include<iostream>
#include<map>
#include<vector>
#include<algorithm>
using namespace std;

int flag; //表示当前在排序哪一学科
vector<map<string, int>> ranks;
struct Student{
    string ID;
    int scores[4];
    int best;
} Stu[2000];
bool cmp(Student s1,Student s2){
    return s1.scores[flag] > s2.scores[flag];  
}

int main(){
    int N, M;
    float temp;
    for (int i = 0; i < 4;i++){
        map<string, int> m;
        ranks.push_back(m);
    }
    char item[4] = {'A','C', 'M', 'E'};
    cin >> N >> M;
    for (int i = 0; i < N;i++){
        Student s;
        cin >> s.ID >> s.scores[1] >> s.scores[2] >> s.scores[3];
        s.scores[0] = (s.scores[1]+s.scores[2]+s.scores[3]) / 3.0+0.5;
        Stu[i] = s;
    }
    for (int i = 0; i < 4;i++){
        flag = i;//表示当前在排序哪一学科
        sort(Stu, Stu + N,cmp);
        int index = 0;//一个学科的排名index从1开始
        temp =Stu[0].scores[i] ;
        ranks[i][Stu[0].ID] = 1;
        index = 1;
        for (int j = 1; j < N;j++){  //将Stu的排序存入ranks中
            if(Stu[j].scores[i]<temp){
                temp = Stu[j].scores[i];
                ranks[i][Stu[j].ID] = index=j+1;
            }
            else if(Stu[j].scores[i]==temp){
                ranks[i][Stu[j].ID] = index;
            }
        }
    }
    string id;
    for (int i = 0; i < M;i++){
        cin >> id;
        int min =ranks[0][id] , index = 0;
        for (int j = 1; j < 4;j++){
            if(ranks[j][id]==0)
                break;
            if(ranks[j][id]<min){
                index = j;
                min = ranks[j][id];
            }
        }
        if(min==0)
            cout << "N/A" ;
        else{
            cout << min << " " << item[index];
        }
        if(i!=M-1)
            printf("\n");
    }
}

```

