## 1011 World Cup Betting (20 分)

With the 2010 FIFA World Cup running, football fans the world over were becoming increasingly excited as the best players from the best teams doing battles for the World Cup trophy in South Africa. Similarly, football betting fans were putting their money where their mouths were, by laying all manner of World Cup bets.

Chinese Football Lottery provided a "Triple Winning" game. The rule of winning was simple: first select any three of the games. Then for each selected game, bet on one of the three possible results -- namely `W` for win, `T` for tie, and `L` for lose. There was an odd assigned to each result. The winner's odd would be the product of the three odds times 65%.

For example, 3 games' odds are given as the following:

```
 W    T    L
1.1  2.5  1.7
1.2  3.1  1.6
4.1  1.2  1.1
```

To obtain the maximum profit, one must buy `W` for the 3rd game, `T` for the 2nd game, and `T` for the 1st game. If each bet takes 2 yuans, then the maximum profit would be (4.1×3.1×2.5×65%−1)×2=39.31 yuans (accurate up to 2 decimal places).

### Input Specification:

Each input file contains one test case. Each case contains the betting information of 3 games. Each game occupies a line with three distinct odds corresponding to `W`, `T` and `L`.

### Output Specification:

For each test case, print in one line the best bet of each game, and the maximum profit accurate up to 2 decimal places. The characters and the number must be separated by one space.

### Sample Input:

```in
1.1 2.5 1.7
1.2 3.1 1.6
4.1 1.2 1.1
```

### Sample Output:

```out
T T W 39.31
```

```C++
#include<iostream>
#include<cstdio>
#include<algorithm>
using namespace std;
int main(){
    float games[3][3];
    char name[3] = {'W', 'T', 'L'};
    int index;
    float profit = 1;
    for (int i = 0; i < 3;i++){
        index = 0;
        for (int j = 0; j < 3;j++){
            cin >> games[i][j];
            if(games[i][j]>games[i][index]){
                index = j;
            }
        }
        cout << name[index]<<" ";
        profit *= games[i][index];
    }
    profit = (profit * 0.65 - 1) * 2;
    printf("%.2f", profit);
}
```

