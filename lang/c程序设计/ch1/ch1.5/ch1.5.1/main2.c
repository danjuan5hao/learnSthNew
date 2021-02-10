#include <stdio.h>
main()
{
    int c;
    while ((c = getchar()) != '/0') {
        putchar(c);
    }
}