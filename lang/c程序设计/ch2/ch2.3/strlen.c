#include <stdio.h>
// #include <string.h>
int strlen(char s[]);

int main()
{
    char site[] = "RUNOOB";
    int lenght = strlen(site);
    printf("%d", lenght);
    return 0;
}

int strlen(char s[])
{
    int p = 0;
    while (s[++p] != '\0')
        ;
    return p;
}