int main()
{
    int n = 011112;
    int b = n & 0177;
    
    return 0;
}

unsigned getbits(unsigned x, int p, int n)
{
    return (x >> (p+1-n)) & ~(~0 << n);
}