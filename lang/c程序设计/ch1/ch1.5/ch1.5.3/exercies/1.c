#include <stdio.h>  
main ()
{
    int c, tc, nc ;
    tc = 0;
    nc = 0;

    while( (c=getchar()) != EOF ){
        if (c == '\n')
            ++nc;
        if (c == '\t')
            ++tc;
    }    
    printf("%d\n", tc);
    printf("%d\n", nc);
}       
