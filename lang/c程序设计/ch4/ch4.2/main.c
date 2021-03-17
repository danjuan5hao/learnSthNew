#include <stdio.h>
#define MAXLINE 100
/* rudimentary calculator */
int getline(char line[], int max);
int main()
{
    double sum, atof(char []);
    char line[MAXLINE];
    sum = 0;
    while (getline(line, MAXLINE) > 0)
        printf("\t%g\n", sum += atof(line));
    return 0;
}