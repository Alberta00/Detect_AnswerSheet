#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main()
{
    int l,r;
    scanf("%d",&l);
    scanf("%d",&r);
    for(int i = l;l<=r;i++){
        if(i % 2 != 0){
            printf("%d\n",i);
        }
    }
}
