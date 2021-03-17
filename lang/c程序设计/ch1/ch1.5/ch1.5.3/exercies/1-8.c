#include <stdio.h>  
int main ()
{
    int c, tc, nc, bc ;
    tc = 0;
    nc = 0;
    bc = 0;

    while( (c=getchar()) != EOF ){
        if (c == '\n')
            ++nc;
        else if (c == '\t')
            ++tc;
        else if (' ' == c)
            ++bc;
    } 

    printf("%d\n", tc);
    printf("%d\n", nc);
    printf("%d\n", bc);
    return 0;
}       
