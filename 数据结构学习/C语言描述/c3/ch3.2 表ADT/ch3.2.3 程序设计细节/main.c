#include <stdio.h>
#include "_List_H.h"

int main()
{
    struct Node header = {0, NULL};
    List l = & header;
    int isEmpty = IsEmpty(l);
    printf("isEmpty: %d", isEmpty);

	int isLast = ISLast(header, l);
	printf("isLast: %d", isLast);

    return 0;
}
