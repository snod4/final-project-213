# final-project-213
Array data structure:
300 Passwords in 4.355 seconds
3000 Passwords in 40.536 seconds
30000 Passwords in 401.785 seconds

Tuning our blocksize:
100 Passwords with,
2 threads per block - 23.085
4 threads per block - 11.579
8 threads per block - 5.963
16 threads per block - 3.120
32 threads per block - 1.580
64 threads per block - 1.124
100 threads per block - 1.388//Not power of two
128 threads per block - 1.077 **Sweetspot
200 threads per block - 1.334//Not power of two
256 threads per block - 1.161

Array data structure with tuned blocksize of 128:
300 Passwords in 2.609
3000 Passwords in 23.136
30000 Passwords in 235.538
