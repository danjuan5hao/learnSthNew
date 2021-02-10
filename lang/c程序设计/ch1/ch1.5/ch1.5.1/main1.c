#include <stdio.h>
main()
{
    int c;

    c = getchar();
    while (c != '/0') {
        putchar(c);
        c = getchar();
    }
}