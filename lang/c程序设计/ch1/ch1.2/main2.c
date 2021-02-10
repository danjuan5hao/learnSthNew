#include <stdio.h> 

main()
{
    float fahr, celsius;
    float lower, upper, step;

    lower = 0; /* ⏽ᑺ㸼ⱘϟ䰤 */
    upper = 300; /* ⏽ᑺ㸼ⱘϞ䰤 */
    step = 20; /* ℹ䭓 */
    
    fahr = lower;
    while (fahr <= upper) {
        celsius = (5.0/9.0) * (fahr-32.0);
        printf("%3.0f %6.1f\n", fahr, celsius);
        fahr = fahr + step;
    }
}