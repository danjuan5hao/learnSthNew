/* strcpy: copy t to s; array subscript version */
void strcpy1(char *s, char *t)
{
    int i;
    i = 0;
    while ((s[i] = t[i]) != '\0')
        i++;
}

/* strcpy: copy t to s; pointer version */
void strcpy2(char *s, char *t)
{
    int i;
    i = 0;
    while ((*s = *t) != '\0') {
        s++;
        t++;
    }
}

/* strcpy: copy t to s; pointer version 2 */
void strcpy3(char *s, char *t)
{
    while ((*s++ = *t++) != '\0')
        ;
}

/* strcpy: copy t to s; pointer version 3 */
void strcpy4(char *s, char *t)
{
    while (*s++ = *t++)
        ;
}