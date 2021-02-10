#include <stdio.h> 

main()
{
    int fahr, celsius;
    int lower, upper, step;

    lower = 0; /* ⏽ᑺ㸼ⱘϟ䰤 */
    upper = 300; /* ⏽ᑺ㸼ⱘϞ䰤 */
    step = 20; /* ℹ䭓 */
    fahr = lower;
    while (fahr <= upper) {
        celsius = 5 * (fahr-32) / 9;
        printf("%d\t%d\n", fahr, celsius);
        fahr = fahr + step;
    }
}