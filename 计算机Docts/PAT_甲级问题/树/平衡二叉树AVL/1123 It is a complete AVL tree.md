## 1123 Is It a Complete AVL Tree (30 分)

An AVL tree is a self-balancing binary search tree. In an AVL tree, the heights of the two child subtrees of any node differ by at most one; if at any time they differ by more than one, rebalancing is done to restore this property. Figures 1-4 illustrate the rotation rules.

| ![F1.jpg](https://images.ptausercontent.com/fb337acb-93b0-4af2-9838-deff5ce98058.jpg) | ![F2.jpg](https://images.ptausercontent.com/d1635de7-3e3f-4aaa-889b-ba29f35890db.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![F3.jpg](https://images.ptausercontent.com/e868e4b9-9fea-4f70-b7a7-1f5d8a3be4ef.jpg) | ![F4.jpg](https://images.ptausercontent.com/98aa1782-cea5-4792-8736-999436cf43a9.jpg) |

Now given a sequence of insertions, you are supposed to output the level-order traversal sequence of the resulting AVL tree, and to tell if it is a complete binary tree.

### Input Specification:

Each input file contains one test case. For each case, the first line contains a positive integer N (≤ 20). Then N distinct integer keys are given in the next line. All the numbers in a line are separated by a space.

### Output Specification:

For each test case, insert the keys one by one into an initially empty AVL tree. Then first print in a line the level-order traversal sequence of the resulting AVL tree. All the numbers in a line must be separated by a space, and there must be no extra space at the end of the line. Then in the next line, print `YES` if the tree is complete, or `NO` if not.

### Sample Input 1:

```in
5
88 70 61 63 65
```

### Sample Output 1:

```out
70 63 88 61 65
YES
```

### Sample Input 2:

```in
8
88 70 61 96 120 90 65 68
```

### Sample Output 2:

```out
88 65 96 61 70 90 120 68
NO
```

**注意：**

1. **AVL建树过程 注意struct结构体中，int height=1，不能为空**
2. **R、L操作内包含更新树高，先更新子结点，再更新父结点**
3. **每一次插入操作后，都需要updateHeight**

```c++
#include<cstdio>
#include<iostream>
#include<queue>
using namespace std;
struct Node{
    Node* left=NULL;Node* right=NULL;
    int data;
    int height=1;
};
int n;
int isComplete=1,after=0;
int getHeight(Node* &root){
    if(root==NULL) return 0;
    return root->height;
}
void updateHeight(Node* &root){
    root->height=max(getHeight(root->left),getHeight(root->right))+1;
}
int getBalanceFactor(Node* &root){
    return getHeight(root->left)-getHeight(root->right);
}
void R(Node* &root){
    Node* temp=root->left;
    root->left=temp->right;
    temp->right=root;
    updateHeight(root);
    updateHeight(temp);
    root=temp;
}
void L(Node* &root){
    Node* temp=root->right;
    root->right=temp->left;
    temp->left=root;
    updateHeight(root);
    updateHeight(temp);
    root=temp;
}
void insert(Node* &root,int data){
    if(root==NULL){
        root=new Node();
        root->data=data;
        return;
    }
    if(data<root->data){
        insert(root->left,data);
        updateHeight(root);
        if(getBalanceFactor(root)==2){
            if(getBalanceFactor(root->left)==1){            //LL型
                R(root);
            }
            else if(getBalanceFactor(root->left)==-1){      //LR型
                L(root->left);
                R(root);
            }
        }
    }
    else if(data>root->data){
        insert(root->right,data);
        updateHeight(root);
        if(getBalanceFactor(root)==-2){
            if(getBalanceFactor(root->right)==-1){
                L(root);
            }
            else if(getBalanceFactor(root->right)==1){
                R(root->right);
                L(root);
            }
        }
    }
}
void levelorder(Node* &root){
    queue<Node*>q;
    q.push(root);
    int cnt=0;
    while(!q.empty()){
        Node* node=q.front();
        q.pop();cnt++;
        printf("%d",node->data);
        if(cnt<n) printf(" ");
        if(node->left!=NULL){
            if(after) isComplete=0;
            q.push(node->left);
        }
        else after=1;
        if(node->right!=NULL){
             if(after) isComplete=0;
            q.push(node->right);
        }
        else after=1;
    }
}
int main(){
    cin>>n;
    int num;
    Node* root=NULL;
    for(int i=0;i<n;i++){
        cin>>num;
        insert(root,num);
    }
    levelorder(root);
    printf(isComplete?"\nYES":"\nNO");
    return 0;
}
```

