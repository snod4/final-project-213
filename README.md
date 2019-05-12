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

Hash table (2-D array) data structure w/ blocksize of 128 & depth of 1000:
300 Passwords in .381
3000 Passwords in 1.080
30000 Passwords in 8.786
100000 Passwords in 117.427

Hash Table (2-D array) data structure with 50000 passwords:
Depth of 10 - 16.741 secods
Depth of 50 - 14.404 seconds
Depth of 100 - 13.997 seconds
Depth of 150 - 14.561 seoncds Broken???
Depth of 250 - 31.950 seconds
Depth of 500 - 37.869 seconds
Depth of 1000 - 41.602 seconds

Hash table (2-D array) data structure w/ blocksize of 128 & tuned depth of 100:
300 Passwords in .351
3000 Passwords in 1.028
30000 Passwords in 8.507
100000 Passwords in 27.922
1000000 Passwords in 277.467 Wow 3604 per second
