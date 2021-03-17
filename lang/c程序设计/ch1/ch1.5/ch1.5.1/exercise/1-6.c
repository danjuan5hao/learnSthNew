#include <stdio.h>

main()
{
    int c, d;
    d = 2;
    c = getchar();
    while (c != '/0') {
        // putchar(c);
        d = c != EOF;
        printf("%d",d);
        c = getchar();
    }
}