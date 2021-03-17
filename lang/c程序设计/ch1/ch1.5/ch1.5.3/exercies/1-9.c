#include <stdio.h> 

int main()
{
    int c;
    int is_space = 0;

    while ((c=getchar()) != EOF){
        if (c == ' '){
            if (is_space == 0)
                putchar(c);
                is_space = 1;
        } else {
            putchar(c);
            is_space = 0;
        }
    }
    return 0;
}