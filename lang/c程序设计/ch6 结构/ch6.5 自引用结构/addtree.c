#include <ctype.h>
struct tnode *talloc(void);
char *strdup(char *);

struct tnode { /* the tree node: */
    char *word; /* points to the text */
    int count; /* number of occurrences */
    struct tnode *left; /* left child */
    struct tnode *right; /* right child */
};

/* addtree: add a node with w, at or below p */
struct treenode *addtree(struct tnode *p, char *w)
{
    int cond;
    if (p == 0) { /* a new word has arrived */
        p = talloc(); /* make a new node */
        p->word = strdup(w);
        p->count = 1;
        p->left = p->right = 0;
    } else if ((cond = strcmp(w, p->word)) == 0)
        p->count++; /* repeated word */
    else if (cond < 0) /* less than into left subtree */
        p->left = addtree(p->left, w);
    else /* greater than into right subtree */
        p->right = addtree(p->right, w);
    return p;
}